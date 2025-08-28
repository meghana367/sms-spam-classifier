import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import re
import string
from utils import preprocess_text

class SpamDetector:
    def __init__(self):
        """Initialize the spam detector with a pre-configured pipeline"""
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        self.is_trained = False
        self.feature_names = None
        self.spam_keywords = [
            'free', 'win', 'winner', 'cash', 'prize', 'urgent', 'call now',
            'limited time', 'act now', 'congratulations', 'selected', 'claim',
            'txt', 'text', 'reply', 'stop', 'offer', 'guaranteed', 'risk free',
            'no obligation', 'click here', 'visit', 'www', 'http', 'link',
            'mobile', 'ringtone', 'download', 'video', 'camera', 'pic',
            '£', '$', 'pound', 'dollar', 'credit', 'loan', 'debt', 'finance',
            '18+', 'adult', 'sex', 'sexy', 'hot', 'babe', 'girl', 'dating',
            'singles', 'meet', 'chat', 'flirt', 'lonely'
        ]
    
    def create_training_data(self):
        """Create synthetic training data based on common spam patterns"""
        spam_messages = [
            "Congratulations! You've won a £1000 cash prize! Call now to claim your reward!",
            "URGENT! Your mobile account will be suspended. Reply STOP to avoid charges.",
            "Free ringtones! Text RING to 12345. £3/week subscription applies.",
            "You've been selected for a special offer! Click here: www.freeprize.com",
            "WINNER! Claim your free iPhone now! Limited time offer expires today!",
            "Hot singles in your area want to meet you! 18+ only. Reply YES",
            "Your loan has been approved! Get cash fast with no credit check required.",
            "FREE entry in 2 a weekly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
            "URGENT! We've been trying to contact u. This is your 2nd notice. Call 09061701461",
            "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575",
            "You have won a Nokia phone! To collect Ur prize call 08712300220 Cost 10p/min",
            "PRIVATE! Your 2003 Account Statement for 07808 has arrived. Call 09066364589",
            "REMINDER FROM O2: To get 2.50 pounds free call credit reply 2 this text with your NAME",
            "Did you hear about the new Divorce Barbie? Yeah, it comes with half of Ken's stuff!",
            "18+ GIRLS WANTED for live adult entertainment. Earn £100s weekly. Text GIRL to 89080",
            "Sunshine Quiz! Win a super Sony DVD recorder worth over £200. Txt PLAY to 83600",
            "CONGRATS U HAVE WON £2000 CASH! To claim call 09050094229 NOW! Cost 50p/min mobile",
            "COSTA HOLIDAYS. Congrats U have won a 4* Tenerife Holiday worth over £1200!",
            "Todays Voda numbers ending with 4311 are selected to receive a £350 reward GUARANTEED",
            "Amazing! Your mobile number has won a £10,000 cash award! To claim call 087147781065"
        ]
        
        ham_messages = [
            "Hey, are you free for lunch today? Let me know!",
            "Thanks for your help with the project. Much appreciated!",
            "Meeting has been moved to 3 PM tomorrow. See you then.",
            "Happy birthday! Hope you have a wonderful day!",
            "Can you pick up milk on your way home? Thanks!",
            "The weather is beautiful today. Perfect for a walk!",
            "Great job on the presentation. Well done!",
            "Don't forget we have dinner plans tonight at 7 PM.",
            "Your package has been delivered to your front door.",
            "Thanks for calling. Talk to you soon!",
            "Good morning! Have a great day at work today.",
            "The report is ready for review. Please check your email.",
            "Reminder: dentist appointment tomorrow at 2 PM.",
            "I'll be running 10 minutes late. See you soon!",
            "Congratulations on your new job! So happy for you.",
            "Can we reschedule our meeting to Friday?",
            "Love you too mom. Thanks for the care package.",
            "The movie was great! Thanks for the recommendation.",
            "Your order is ready for pickup at the store.",
            "See you at the gym tomorrow morning at 7 AM."
        ]
        
        # Create DataFrame
        messages = spam_messages + ham_messages
        labels = ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages)
        
        return pd.DataFrame({
            'message': messages,
            'label': labels
        })
    
    def train_model(self):
        """Train the spam detection model"""
        # Create training data
        df = self.create_training_data()
        
        # Preprocess messages
        df['processed_message'] = df['message'].apply(preprocess_text)
        
        # Train the pipeline
        X = df['processed_message']
        y = df['label']
        
        self.pipeline.fit(X, y)
        self.is_trained = True
        
        # Store feature names for explanation
        self.feature_names = self.pipeline['tfidf'].get_feature_names_out()
        
        return self
    
    def predict(self, message):
        """
        Predict if a message is spam or not
        
        Args:
            message (str): SMS message to classify
            
        Returns:
            tuple: (prediction, confidence, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess the message
        processed_message = preprocess_text(message)
        
        # Get prediction and probabilities
        prediction = self.pipeline.predict([processed_message])[0]
        probabilities = self.pipeline.predict_proba([processed_message])[0]
        
        # Calculate confidence (max probability)
        confidence = max(probabilities)
        
        return prediction, confidence, probabilities
    
    def explain_prediction(self, message, top_n=10):
        """
        Explain the prediction by showing top contributing features
        
        Args:
            message (str): Message to explain
            top_n (int): Number of top features to return
            
        Returns:
            list: List of (feature, score) tuples
        """
        if not self.is_trained or self.feature_names is None:
            return []
        
        # Get TF-IDF features
        processed_message = preprocess_text(message)
        tfidf_vector = self.pipeline['tfidf'].transform([processed_message])
        
        # Get feature scores
        feature_scores = tfidf_vector.toarray()[0]
        
        # Get top features
        feature_indices = np.argsort(feature_scores)[::-1][:top_n]
        top_features = []
        
        for idx in feature_indices:
            if feature_scores[idx] > 0 and idx < len(self.feature_names):
                feature_name = self.feature_names[idx]
                score = feature_scores[idx]
                top_features.append((feature_name, score))
        
        return top_features
    
    def get_top_spam_features(self, top_n=20):
        """Get the most important features for spam classification"""
        if not self.is_trained or self.feature_names is None:
            return self.spam_keywords[:top_n]
        
        # Get feature coefficients from the classifier
        classifier = self.pipeline['classifier']
        feature_scores = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
        
        # Get top features
        top_indices = np.argsort(feature_scores)[::-1][:top_n]
        top_features = [self.feature_names[idx] for idx in top_indices if idx < len(self.feature_names)]
        
        return top_features
    
    def get_feature_importance(self, top_n=50):
        """Get feature importance scores"""
        if not self.is_trained or self.feature_names is None:
            return {}
        
        # Calculate feature importance
        classifier = self.pipeline['classifier']
        feature_scores = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
        
        # Get top features with scores
        top_indices = np.argsort(feature_scores)[::-1][:top_n]
        feature_importance = {}
        
        for idx in top_indices:
            if idx < len(self.feature_names):
                feature_name = self.feature_names[idx]
                score = feature_scores[idx]
                feature_importance[feature_name] = score
        
        return feature_importance
    
    def get_model_stats(self):
        """Get model statistics"""
        if not self.is_trained:
            return {"Status": "Not Trained"}
        
        # Create test data for cross-validation
        df = self.create_training_data()
        df['processed_message'] = df['message'].apply(preprocess_text)
        
        # Perform cross-validation
        scores = cross_val_score(
            self.pipeline, 
            df['processed_message'], 
            df['label'], 
            cv=3, 
            scoring='accuracy'
        )
        
        return {
            "Status": "Trained",
            "Features": len(self.feature_names) if self.feature_names is not None else 0,
            "Accuracy": f"{scores.mean():.2%}",
            "Training Samples": len(df)
        }
    
    def batch_predict(self, messages):
        """
        Predict multiple messages at once
        
        Args:
            messages (list): List of messages to classify
            
        Returns:
            list: List of prediction results
        """
        results = []
        for message in messages:
            prediction, confidence, probabilities = self.predict(message)
            results.append({
                'message': message,
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities
            })
        return results
