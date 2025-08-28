import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess SMS text for spam detection
    
    Args:
        text (str): Input text message
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{10,11}\b', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing spaces
    text = text.strip()
    
    return text

def extract_features(text):
    """
    Extract additional features from text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of extracted features
    """
    if not isinstance(text, str):
        text = ""
    
    features = {}
    
    # Basic text statistics
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['char_count'] = len(text)
    
    # Special character counts
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['dollar_count'] = text.count('$') + text.count('£')
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    
    # Digit statistics
    features['digit_count'] = sum(1 for c in text if c.isdigit())
    features['digit_ratio'] = features['digit_count'] / max(len(text), 1)
    
    # URL and phone number indicators
    features['has_url'] = 1 if re.search(r'http|www|\.com', text.lower()) else 0
    features['has_phone'] = 1 if re.search(r'\d{10,11}', text) else 0
    
    return features

def get_sample_messages():
    """
    Get sample messages for testing
    
    Returns:
        dict: Dictionary with spam and ham sample messages
    """
    samples = {
        'spam': [
            "URGENT! You've won £500 cash! Call 09876543210 now to claim. Txt STOP to opt out.",
            "FREE iPhone! You've been selected as today's winner. Click www.freephone.com",
            "Hot singles near you! 18+ only. Reply HOT to chat now. £2.50/msg",
            "Your loan of £5000 has been approved! No credit check. Call now 08001234567",
            "CONGRATULATIONS! You've won a £1000 shopping voucher. Text WIN to 12345"
        ],
        'ham': [
            "Hey, are you coming to the party tonight? Let me know!",
            "Thanks for your help today. Really appreciate it!",
            "Meeting moved to 2 PM tomorrow. Conference room B.",
            "Can you pick up some groceries? Need milk and bread.",
            "Happy anniversary! Love you so much. See you tonight."
        ]
    }
    
    return samples

def calculate_spam_score(text):
    """
    Calculate a simple spam score based on keywords
    
    Args:
        text (str): Input text
        
    Returns:
        float: Spam score between 0 and 1
    """
    spam_keywords = [
        'free', 'win', 'winner', 'cash', 'prize', 'urgent', 'call now',
        'limited time', 'act now', 'congratulations', 'selected', 'claim',
        'txt', 'text', 'reply', 'stop', 'offer', 'guaranteed', 'risk free',
        'no obligation', 'click here', 'visit', 'www', 'http', 'link',
        '£', '$', 'pound', 'dollar', 'credit', 'loan', 'debt',
        '18+', 'adult', 'singles', 'meet', 'chat'
    ]
    
    if not isinstance(text, str):
        return 0.0
    
    text_lower = text.lower()
    keyword_count = sum(1 for keyword in spam_keywords if keyword in text_lower)
    
    # Normalize by text length and keyword list length
    score = keyword_count / max(len(spam_keywords), 1)
    
    # Additional factors
    if re.search(r'\d{10,11}', text):  # Phone numbers
        score += 0.2
    if text.count('!') > 2:  # Multiple exclamations
        score += 0.1
    if re.search(r'http|www', text.lower()):  # URLs
        score += 0.15
    
    return min(score, 1.0)

def validate_message(message):
    """
    Validate input message
    
    Args:
        message (str): Input message
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not message:
        return False, "Message cannot be empty"
    
    if not isinstance(message, str):
        return False, "Message must be a string"
    
    if len(message.strip()) == 0:
        return False, "Message cannot be just whitespace"
    
    if len(message) > 1000:
        return False, "Message is too long (max 1000 characters)"
    
    return True, ""

def format_confidence(confidence):
    """
    Format confidence score for display
    
    Args:
        confidence (float): Confidence score
        
    Returns:
        str: Formatted confidence string
    """
    if confidence >= 0.9:
        return f"{confidence:.1%} (Very High)"
    elif confidence >= 0.7:
        return f"{confidence:.1%} (High)"
    elif confidence >= 0.5:
        return f"{confidence:.1%} (Medium)"
    else:
        return f"{confidence:.1%} (Low)"
