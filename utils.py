# app/utils.py
import re
import string
import nltk
from nltk.corpus import stopwords
import logging

logger = logging.getLogger(__name__)

# Download stopwords on first run
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Filipino stopwords from the notebook
fil_stopwords = set([
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
            ])

# Combine English and Filipino stopwords
all_stopwords = set(stopwords.words("english")).union(fil_stopwords)

def clean_text(text, debug=False):
    """
    Clean text following the notebook preprocessing pipeline.
    Set debug=True to see before/after comparison.
    
    This creates the 'preprocessed_text' column used for sentiment analysis.
    
    Args:
        text: Raw text to preprocess
        debug: If True, logs before/after comparison
        
    Returns:
        Preprocessed text (lowercase, no URLs, no stopwords, letters only)
    """
    original = str(text)
    
    # Step 1: Lowercase
    text = original.lower()
    
    # Step 2: Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Step 3: Keep only letters and accented characters (removes numbers and punctuation)
    text = re.sub(r"[^a-zA-Záéíóúüñ\s]", "", text)
    
    # Step 4: Remove special characters (redundant but kept for notebook consistency)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Step 5: Remove stopwords
    text = " ".join([w for w in text.split() if w not in all_stopwords])
    
    if debug:
        logger.info(f"BEFORE: {original}")
        logger.info(f"AFTER (PREPROCESSED): {text}")
        logger.info(f"Words removed: {len(original.split()) - len(text.split())}")
        logger.info("-" * 50)
    
    return text