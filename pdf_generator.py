#pdf_generator/py
from dotenv import load_dotenv
load_dotenv()
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import base64
import re
from collections import Counter
import requests
import json
import logging
import os
import google.generativeai as genai

router = APIRouter()
logger = logging.getLogger(__name__)

filipino_stopwords = {
                'po', 'opo', 'sa', 'nang', 'ng', 'ang', 'lng', 'lang', 'ba', 'yung', 'ung',
                'na', 'ay', 'si', 'ni', 'kay', 'para', 'mga', 'at', 'o', 'kaya', 'pero',
                'kung', 'kapag', 'habang', 'dahil', 'kasi', 'naman', 'din', 'rin', 'daw',
                'raw', 'sabi', 'kuno', 'dito', 'doon', 'diyan', 'ito', 'yan', 'yun',
                'ko', 'sana', 'pa', 'ako', 'wala', 'hindi', 'nyo', 'mag',
                'nag', 'kami', 'pag', 'namin', 'nila', 'di', 'nga', 'kahit',
                'lahat', 'walang', 'kayo', 'nmn', 'tapos', 'isang', 'ka',
                'sila', 'pwede', 'pano', 'paano', 'bakit', 'wla', 'ma',
                'araw', 'nman', 'kc', 'parin', 'un', 'hanggang',
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'must', 'this', 'that', 'these', 'those',
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
                'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                'own', 'same', 'so', 'than', 'too', 'in', 'very', 's', 't', 'can', 'will', 'just',
                'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
                'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
    }

def call_gemini_api(prompt, model_name="gemini-flash-latest", timeout=60):
    """Call the Google Gemini API for text generation."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY environment variable not set.")
            return None
        
        genai.configure(api_key=api_key)
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        model = genai.GenerativeModel(model_name)
        
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            request_options={'timeout': timeout}
        )
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error connecting to Google Gemini API: {str(e)}")
        return None

def generate_ai_insights(data_summary):
    """Generates insights by sending a data summary to Gemini."""
    
    prompt = f"""
You are a helpful data reporter. Your audience is a group of senior managers who are not tech-savvy. 
Your goal is to explain what the sentiment data means in a clear, detailed, and simple-to-understand report.
Do not just list the numbers; explain their importance.

**Here is the data summary you must analyze:**
- **Agency:** {data_summary.get('agency', 'N/A')}
- **Time Period:** {data_summary.get('start_date', 'N/A')} to {data_summary.get('end_date', 'N/A')}
- **Total Comments Analyzed:** {data_summary.get('total_comments', 0)}

**Sentiment Breakdown:**
- **Positive:** {data_summary.get('positive_count', 0)} ({data_summary.get('positive_percent', 0):.1f}%)
- **Neutral:** {data_summary.get('neutral_count', 0)} ({data_summary.get('neutral_percent', 0):.1f}%)
- **Negative:** {data_summary.get('negative_count', 0)} ({data_summary.get('negative_percent', 0):.1f}%)
- **Overall Sentiment:** The majority of comments ({data_summary.get('top_sentiment', 'N/A')}) were the largest group.

**Key Observations:**
- **Busiest Day for Feedback:** {data_summary.get('busiest_day', 'N/A')}
- **Top Discussion Keywords:** {', '.join(data_summary.get('common_words', ['N/A']))}

**--- YOUR REPORT ---**

**Your Task:** Write a detailed, paragraph-by-paragraph report answering the following questions for your audience.

**1. Executive Summary (The Big Picture):**
Start with a simple, high-level summary. What is the overall story of the feedback for this agency? (e.g., "For the {data_summary.get('agency', 'N/A')}, the public feedback is largely neutral, but there is a significant group of negative comments we need to look at.")

**2. The Overall Public Mood (Sentiment Explained):**
Explain the sentiment numbers in detail.
* What does it mean that **{data_summary.get('top_sentiment', 'N/A')}** is the biggest group? A high neutral count, for example, often means people are asking questions (like "where," "how," "pano") rather than expressing opinions.
* Look at the balance between Positive ({data_summary.get('positive_percent', 0):.1f}%) and Negative ({data_summary.get('negative_percent', 0):.1f}%). Is this ratio healthy or concerning? Explain why.

**3. What Are People Talking About? (Top Keywords):**
List the main keywords: **{', '.join(data_summary.get('common_words', ['N/A']))}**.
Explain what these words likely mean *in context* for this specific agency. (For example, if the agency is DOLE and the top word is 'sahod' (salary), it clearly means most conversations are about pay).

**4. When Are People Reaching Out? (Busiest Day):**
The busiest day for comments was **{data_summary.get('busiest_day', 'N/A')}**.
Explain what this might imply. (e.g., "This could be because of a weekly social media post, or perhaps it's the day before a payroll or deadline, leading to more questions.")

**5. Simple Recommendations (What to do next?):**
Based *only* on this data, what are 1-2 simple, actionable next steps for the managers? (e.g., "We should investigate *why* {data_summary.get('busiest_day', 'N/A')} is so busy to prepare our staff," or "We should look closer at the negative comments that mention the keyword '{data_summary.get('common_words', ['N/A'])[0]}'.")

**Tone:** Be clear, helpful, and explanatory. Do not use complex jargon.
"""
    ai_response = call_gemini_api(prompt, model_name="gemini-flash-latest")
    return ai_response

class PDFReportRequest(BaseModel):
    data: List[Dict[str, Any]]
    config: Dict[str, Any]

def create_sentiment_distribution_chart(df):
    """Create sentiment metrics visualization as pie chart"""
    sentiment_counts = df['sentiment_label'].value_counts()
    for label in ['Positive', 'Neutral', 'Negative']:
        if label not in sentiment_counts:
            sentiment_counts[label] = 0
    sentiment_counts = sentiment_counts[sentiment_counts > 0]
    sentiment_counts = sentiment_counts.reindex(['Positive', 'Neutral', 'Negative']).dropna()

    fig, ax = plt.subplots(figsize=(6, 6))
    colors_map = {'Positive': '#10b981', 'Neutral': '#f59e0b', 'Negative': '#ef4444'}
    colors_list = [colors_map.get(label, 'gray') for label in sentiment_counts.index]
    
    wedges, texts, autotexts = ax.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        colors=colors_list,
        autopct='%1.1f%%',
        startangle=90
    )
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
    for text in texts:
        text.set_fontsize(12)
        text.set_weight('bold')
    ax.set_title('Sentiment Distribution', fontsize=14, weight='bold')
    ax.axis('equal')
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    img_buffer.seek(0)
    return img_buffer

def create_weekly_sentiment_chart(df):
    """Create weekly sentiment trend visualization"""
    if df.empty: return None
    df['comment_timestamp'] = pd.to_datetime(df['comment_timestamp'])
    df['day_of_week'] = df['comment_timestamp'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    pivot_data = df.groupby(['day_of_week', 'sentiment_label']).size().unstack(fill_value=0)
    pivot_data = pivot_data.reindex(day_order)
    
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        if sentiment not in pivot_data.columns:
            pivot_data[sentiment] = 0
    pivot_data = pivot_data[['Positive', 'Neutral', 'Negative']]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {'Positive': '#10b981', 'Neutral': '#f59e0b', 'Negative': '#ef4444'}
    pivot_data.plot(kind='bar', stacked=False, ax=ax, 
                    color=[colors[col] for col in pivot_data.columns], width=0.8)
    
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Number of Comments')
    ax.set_title('Weekly Sentiment Trend')
    ax.legend(title='Sentiment', loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    img_buffer.seek(0)
    return img_buffer

def create_monthly_sentiment_chart(df):
    """Create monthly sentiment trend visualization"""
    if df.empty: return None
    df['comment_timestamp'] = pd.to_datetime(df['comment_timestamp'])
    df['month'] = df['comment_timestamp'].dt.to_period('M')
    monthly_data = df.groupby(['month', 'sentiment_label']).size().unstack(fill_value=0)
    
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        if sentiment not in monthly_data.columns:
            monthly_data[sentiment] = 0
    monthly_data = monthly_data[['Positive', 'Neutral', 'Negative']]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x_dates = monthly_data.index.to_timestamp()
    y_pos = monthly_data['Positive']
    y_neu = y_pos + monthly_data['Neutral']
    y_neg = y_neu + monthly_data['Negative']
    
    ax.fill_between(x_dates, 0, y_pos, color='#10b981', alpha=0.7, label='Positive')
    ax.fill_between(x_dates, y_pos, y_neu, color='#f59e0b', alpha=0.7, label='Neutral')
    ax.fill_between(x_dates, y_neu, y_neg, color='#ef4444', alpha=0.7, label='Negative')
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Comments')
    ax.set_title('Monthly Sentiment Trend')
    ax.set_ylim(bottom=0)
    ax.legend(title='Sentiment', loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    img_buffer.seek(0)
    return img_buffer

def create_wordcloud_image(df, sentiment):
    """Create word cloud visualization"""
    text_column = 'preprocessed_text' if 'preprocessed_text' in df.columns else 'comment_text'
    text_data = " ".join(df[df['sentiment_label'] == sentiment][text_column].astype(str))
    if not text_data.strip(): return None
    
    text = re.sub(r'[^\w\s]', ' ', text_data.lower())
    words = text.split()
    filtered_words = [word for word in words if len(word) > 2 and word not in filipino_stopwords]
    word_counts = Counter(filtered_words)
    if not word_counts: return None
    
    color_palette = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#0ea5e9', '#d946ef', '#eab308', '#f43f5e', '#22d3ee']
    def random_color_func(**kwargs): return kwargs['random_state'].choice(color_palette)

    wordcloud = WordCloud(
        width=800, height=300, background_color="white",
        color_func=random_color_func, collocations=False, max_words=100, prefer_horizontal=1.0
    ).generate_from_frequencies(word_counts)
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'{sentiment} Sentiment Word Cloud')
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    img_buffer.seek(0)
    return img_buffer

@router.post("/generate_pdf_report")
async def generate_pdf_report(request: PDFReportRequest):
    """Generate PDF report with visualizations"""
    try:
        df = pd.DataFrame(request.data)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data provided for report generation")
        
        df['comment_timestamp'] = pd.to_datetime(df['comment_timestamp'])
        config = request.config
        
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                                rightMargin=0.5*inch, leftMargin=0.5*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        elements = []
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#1e3a8a'), spaceAfter=30, alignment=TA_CENTER)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#05004E'), spaceAfter=12, spaceBefore=12)
        
        # 1. Title
        title = Paragraph("Sentiment Analysis Report", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))
        
        # 2. Info
        info_text = f"""
        <b>Agency:</b> {config['agency']}<br/>
        <b>Period:</b> {config['start_date']} to {config['end_date']}<br/>
        <b>Total Comments:</b> {len(df)}<br/>
        <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        info = Paragraph(info_text, styles['Normal'])
        elements.append(info)
        elements.append(Spacer(1, 0.3*inch))
        
        # 3. Metrics (MOVED UP)
        if config.get('include_metrics', True):
            elements.append(Paragraph("Sentiment Metrics", heading_style))
            
            sentiment_counts = df['sentiment_label'].value_counts()
            total = len(df)
            metrics_data = [
                ['Sentiment', 'Count', 'Percentage'],
                ['Positive', sentiment_counts.get('Positive', 0), f"{(sentiment_counts.get('Positive', 0)/total*100):.1f}%"],
                ['Neutral', sentiment_counts.get('Neutral', 0), f"{(sentiment_counts.get('Neutral', 0)/total*100):.1f}%"],
                ['Negative', sentiment_counts.get('Negative', 0), f"{(sentiment_counts.get('Negative', 0)/total*100):.1f}%"]
            ]
            metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#131C55')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            elements.append(metrics_table)
            elements.append(Spacer(1, 0.3*inch))
            
            metrics_img = create_sentiment_distribution_chart(df) 
            if metrics_img:
                img = Image(metrics_img, width=5*inch, height=5*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3*inch))
        
        # 4. Weekly Trends (MOVED UP)
        if config.get('include_trends', True):
            elements.append(PageBreak())
            elements.append(Paragraph("Weekly Sentiment Trend", heading_style))
            weekly_img = create_weekly_sentiment_chart(df)
            if weekly_img:
                img = Image(weekly_img, width=7*inch, height=3.5*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3*inch))
        
        # 5. Monthly Trends (MOVED UP)
        if config.get('include_overall_trend', True):
            if not config.get('include_trends', True): elements.append(PageBreak())
            elements.append(Paragraph("Sentiment Trend Over Time", heading_style))
            monthly_img = create_monthly_sentiment_chart(df)
            if monthly_img:
                img = Image(monthly_img, width=7*inch, height=3.5*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3*inch))
        
        # 6. Word Cloud (MOVED UP)
        if config.get('include_wordcloud', True):
            elements.append(PageBreak())
            elements.append(Paragraph("Word Clouds", heading_style))
            for sentiment in ['Positive', 'Neutral', 'Negative']:
                if sentiment in df['sentiment_label'].values:
                    wc_img = create_wordcloud_image(df, sentiment)
                    if wc_img:
                        elements.append(Spacer(1, 0.2*inch))
                        img = Image(wc_img, width=7*inch, height=2.625*inch)
                        elements.append(img)
                        elements.append(Spacer(1, 0.2*inch))
        
        # 7. AI Insights (MOVED DOWN)
        try:            
            sentiment_counts = df['sentiment_label'].value_counts()
            total = int(len(df))
            pos = int(sentiment_counts.get('Positive', 0))
            neu = int(sentiment_counts.get('Neutral', 0))
            neg = int(sentiment_counts.get('Negative', 0))
            top_sentiment = max((('Positive', pos), ('Neutral', neu), ('Negative', neg)), key=lambda x: x[1])[0]

            busiest_day = 'N/A'
            try:
                dow_counts = df['comment_timestamp'].dt.day_name().value_counts()
                if not dow_counts.empty: busiest_day = dow_counts.idxmax()
            except: pass

            common_words = []
            try:
                text_series = df.get('preprocessed_text', df.get('comment_text', pd.Series([], dtype=str))).astype(str)
                joined = " ".join(text_series.tolist()).lower()
                tokens = [t for t in re.split(r"[^a-z0-9]+", joined) if len(t) > 3 and t not in filipino_stopwords]
                common = [w for w,_ in Counter(tokens).most_common(5)]
                if common: common_words = common
            except: pass

            data_summary = {
                "agency": config.get('agency','All Agency'),
                "start_date": config.get('start_date','N/A'),
                "end_date": config.get('end_date','N/A'),
                "total_comments": total,
                "positive_count": pos,
                "neutral_count": neu,
                "negative_count": neg,
                "positive_percent": (pos/total*100) if total > 0 else 0,
                "neutral_percent": (neu/total*100) if total > 0 else 0,
                "negative_percent": (neg/total*100) if total > 0 else 0,
                "top_sentiment": top_sentiment.lower(),
                "busiest_day": busiest_day,
                "common_words": common_words
            }
            
            logger.info(f"Generating AI insights with Google Gemini for agency: {data_summary['agency']}")
            ai_insight_text = generate_ai_insights(data_summary)
            
            elements.append(Paragraph("AI-Generated Insights", heading_style))
            
            if ai_insight_text:
                formatted_ai_text = ai_insight_text
                formatted_ai_text = re.sub(r'^(#### |### |## )(.*?)$', r'<b>\2</b>', formatted_ai_text, flags=re.MULTILINE)
                formatted_ai_text = re.sub(r'^\* (.*?)$', r'&bull; &nbsp;\1', formatted_ai_text, flags=re.MULTILINE)
                formatted_ai_text = re.sub(r'^(\d+\.) (.*?)$', r'&nbsp;&nbsp;&nbsp;&nbsp;\1 \2', formatted_ai_text, flags=re.MULTILINE)
                formatted_ai_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_ai_text)
                formatted_ai_text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', formatted_ai_text)
                formatted_ai_text = re.sub(r'^\s*\*\s*$', '', formatted_ai_text, flags=re.MULTILINE)
                formatted_ai_text = re.sub(r'^\s*\*\*\*\s*$', '', formatted_ai_text, flags=re.MULTILINE)
                formatted_ai_text = re.sub(r'\|', '', formatted_ai_text)
                formatted_ai_text = re.sub(r'^[:\- ]+$', '', formatted_ai_text, flags=re.MULTILINE)
                formatted_ai_text = formatted_ai_text.replace('\n', '<br/>')
                formatted_ai_text = re.sub(r'(<br/>\s*){2,}', '<br/><br/>', formatted_ai_text)

                elements.append(Paragraph(formatted_ai_text, styles['BodyText']))
            else:
                # Fallback
                insight_lines = []
                insight_lines.append(f"We analyzed {total} comments for {config.get('agency','All Agency')} between {config.get('start_date','N/A')} and {config.get('end_date','N/A')}.")
                insight_lines.append(f"Most comments were {top_sentiment.lower()} in tone.")
                if total > 0:
                    insight_lines.append(f"Breakdown: Positive {pos/total*100:.1f}%, Neutral {neu/total*100:.1f}%, Negative {neg/total*100:.1f}%.")
                if busiest_day != 'N/A':
                    insight_lines.append(f"Engagement peaked on {busiest_day}s.")
                if common_words:
                    insight_lines.append(f"Common discussion terms included: {', '.join(common_words)}.")
                
                elements.append(Paragraph(" ".join(insight_lines), styles['BodyText']))
            elements.append(Spacer(1, 0.3*inch))
        except Exception as e:
            logger.error(f"Error during AI insight generation block: {e}", exc_info=True)
            elements.append(Paragraph("Insights generation failed.", styles['BodyText']))
            pass 

        # 8. Notes (Last)
        notes_text = config.get('_notes', '').strip()
        if notes_text:
            elements.append(Paragraph("Notes/Descriptions", heading_style))
            formatted_notes = notes_text.replace('\n', '<br/>')
            notes_paragraph = Paragraph(formatted_notes, styles['BodyText'])
            elements.append(notes_paragraph)
        
        doc.build(elements)
        pdf_buffer.seek(0)
        
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=sentiment_report_{config['start_date']}_{config['end_date']}.pdf"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

# Helper functions for individual insights (kept for reference if needed, though unused by main generator now)
def get_word_cloud_insights(data: pd.DataFrame, sentiment: str) -> (str, list):
    sentiment_data = data[data['sentiment_label'] == sentiment]
    if sentiment_data.empty: return f"No {sentiment.lower()} comments were found.", []
    text_series = sentiment_data['preprocessed_text'].fillna(sentiment_data['comment_text'])
    full_text = ' '.join(text_series.astype(str))
    words = re.findall(r'\b\w{3,}\b', full_text.lower())
    word_counts = pd.Series(words).value_counts()
    top_words = word_counts.head(5).index.tolist()
    word_list = word_counts.head(75).reset_index().values.tolist()
    insight = f"The most prominent {sentiment.lower()} topics are: "
    if len(top_words) > 1: insight += f"{', '.join(top_words[:-1])}, and {top_words[-1]}."
    elif len(top_words) == 1: insight += f"{top_words[0]}."
    else: insight = f"No significant {sentiment.lower()} keywords were found."
    return insight, word_list

def get_distribution_insights(data: pd.DataFrame) -> str:
    if data.empty: return "No data available."
    counts = data['sentiment_label'].value_counts()
    total = counts.sum()
    if total == 0: return "No comments found."
    majority_sentiment = counts.idxmax()
    majority_percent = (counts.max() / total) * 100
    insight = f"The sentiment is predominantly {majority_sentiment}, making up {majority_percent:.0f}% of all comments."
    return insight

def get_trend_insights(data: pd.DataFrame) -> str:
    if data.empty: return "No data available."
    df = data.copy()
    df['date'] = pd.to_datetime(df['comment_timestamp'])
    df = df.set_index('date').sort_index()
    if df.empty: return "No valid timestamps found."
    monthly_sentiment = df.groupby([pd.Grouper(freq='M'), 'sentiment_label']).size().unstack(fill_value=0)
    if monthly_sentiment.empty: return "Not enough data."
    if 'Negative' in monthly_sentiment.columns:
        peak_neg_month = monthly_sentiment['Negative'].idxmax().strftime('%B %Y')
        peak_neg_count = monthly_sentiment['Negative'].max()
        insight = f"Negative sentiment peaked in {peak_neg_month} with {peak_neg_count} comments."
    else:
        insight = "No significant negative sentiment was recorded."
    if 'Positive' in monthly_sentiment.columns:
        peak_pos_month = monthly_sentiment['Positive'].idxmax().strftime('%B %Y')
        peak_pos_count = monthly_sentiment['Positive'].max()
        insight += f" Positive sentiment was highest in {peak_pos_month} ({peak_pos_count} comments)."
    return insight

