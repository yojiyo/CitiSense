# app/model_loader.py
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import re
import os
import logging
from utils import clean_text

logger = logging.getLogger(__name__)

def load_model(model_name_or_path="bert-base-uncased"):
    """Load BERT model and tokenizer."""
    logger.info("Loading model and tokenizer...")
    try:
        # Load the tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        
        # Load the TensorFlow model
        model = TFBertForSequenceClassification.from_pretrained(model_name_or_path)
        
        logger.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model. Error: {e}")
        return None, None

def process_text(text):
    """
    Process text for sentiment analysis using the same preprocessing
    as the training pipeline.
    
    Args:
        text: Raw text to process
        
    Returns:
        Cleaned text ready for model input
    """
    return clean_text(text)

def analyze_sentiment(text, model, tokenizer):
    """
    Analyze sentiment of text using the loaded model.
    
    Args:
        text: Raw text to analyze
        model: Loaded BERT model
        tokenizer: Loaded BERT tokenizer
        
    Returns:
        Sentiment label as string ('Positive', 'Neutral', 'Negative')
    """
    if not model or not tokenizer:
        logger.error("Model or tokenizer not loaded")
        return "Model not loaded"

    # Apply preprocessing
    cleaned_text = process_text(text)
    
    # If text becomes empty after cleaning, return neutral
    if not cleaned_text.strip():
        logger.warning(f"Text became empty after cleaning: {text[:50]}...")
        return "Neutral"

    try:
        # Tokenize
        inputs = tokenizer(
            cleaned_text,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='tf'
        )

        # Get model predictions
        outputs = model(inputs)
        probabilities = tf.nn.softmax(outputs.logits, axis=-1)
        sentiment_index = tf.argmax(probabilities, axis=1).numpy()[0]
        
        # Map to your label convention (from notebook: 0=Neutral, 1=Positive, 2=Negative)
        sentiment_labels = {
            0: 'Neutral',
            1: 'Positive',
            2: 'Negative'
        }
        
        result = sentiment_labels.get(sentiment_index, 'Unknown')
        logger.debug(f"Sentiment for '{cleaned_text[:30]}...': {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
        return "Error"