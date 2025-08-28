import streamlit as st
import pandas as pd
import numpy as np
from model import SpamDetector
from utils import preprocess_text, get_sample_messages
import plotly.express as px
import plotly.graph_objects as go

# Initialize the spam detector
@st.cache_resource
def load_model():
    """Load and cache the spam detection model"""
    detector = SpamDetector()
    detector.train_model()
    return detector

def main():
    st.set_page_config(
        page_title="SMS Spam Detector",
        page_icon="📱",
        layout="wide"
    )
    
    # Header with images
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("📱 SMS Spam Detector")
        st.markdown("### Protect yourself from spam messages using AI")
    
    # Load model
    try:
        detector = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("📊 About")
        st.markdown("""
        This app uses machine learning to detect spam SMS messages.
        
        **Features:**
        - Real-time spam detection
        - Confidence scoring
        - Batch processing
        - Model explanations
        
        **Technology:**
        - TF-IDF Vectorization
        - Naive Bayes Classification
        - scikit-learn
        """)
        
        st.header("⚠️ Spam Indicators")
        spam_indicators = detector.get_top_spam_features()
        for feature in spam_indicators[:10]:
            st.markdown(f"• {feature}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Message", "📋 Batch Processing", "🧪 Sample Tests", "📈 Model Info"])
    
    with tab1:
        st.header("Analyze Single SMS Message")
        
        # Text input
        message = st.text_area(
            "Enter SMS message to analyze:",
            height=100,
            placeholder="Type your SMS message here..."
        )
        
        if st.button("🔍 Analyze Message", type="primary"):
            if message.strip():
                try:
                    # Get prediction
                    prediction, confidence, probabilities = detector.predict(message)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 'spam':
                            st.error(f"🚨 **SPAM DETECTED**")
                            st.metric("Spam Probability", f"{probabilities[1]:.2%}")
                        else:
                            st.success(f"✅ **NOT SPAM**")
                            st.metric("Ham Probability", f"{probabilities[0]:.2%}")
                    
                    with col2:
                        st.metric("Confidence Score", f"{confidence:.2%}")
                        
                        # Confidence visualization
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = confidence * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Confidence"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "red" if prediction == 'spam' else "green"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "lime"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature analysis
                    st.subheader("🔍 Message Analysis")
                    
                    # Show key features
                    key_features = detector.explain_prediction(message)
                    if key_features:
                        st.write("**Key features detected:**")
                        feature_df = pd.DataFrame(key_features)
                        feature_df.columns = ['Feature', 'Score']
                        st.dataframe(feature_df, use_container_width=True)
                    
                    # Message statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Characters", len(message))
                    with col2:
                        st.metric("Words", len(message.split()))
                    with col3:
                        st.metric("Uppercase %", f"{sum(1 for c in message if c.isupper()) / len(message) * 100:.1f}%")
                    with col4:
                        exclamation_count = message.count('!')
                        st.metric("Exclamations", exclamation_count)
                        
                except Exception as e:
                    st.error(f"Error analyzing message: {str(e)}")
            else:
                st.warning("Please enter a message to analyze.")
    
    with tab2:
        st.header("Batch Processing")
        st.markdown("Upload multiple SMS messages for bulk analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file with SMS messages",
            type=['csv'],
            help="CSV should have a column named 'message' containing SMS texts"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'message' not in df.columns:
                    st.error("CSV file must contain a 'message' column")
                else:
                    st.success(f"Loaded {len(df)} messages")
                    
                    if st.button("🔍 Analyze All Messages", type="primary"):
                        progress_bar = st.progress(0)
                        results = []
                        
                        for i, message in enumerate(df['message']):
                            if pd.isna(message):
                                continue
                                
                            prediction, confidence, probabilities = detector.predict(str(message))
                            results.append({
                                'message': message,
                                'prediction': prediction,
                                'confidence': confidence,
                                'spam_probability': probabilities[1],
                                'ham_probability': probabilities[0]
                            })
                            
                            progress_bar.progress((i + 1) / len(df))
                        
                        results_df = pd.DataFrame(results)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            spam_count = len(results_df[results_df['prediction'] == 'spam'])
                            st.metric("Spam Messages", spam_count)
                        with col2:
                            ham_count = len(results_df[results_df['prediction'] == 'ham'])
                            st.metric("Ham Messages", ham_count)
                        with col3:
                            avg_confidence = results_df['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                        
                        # Visualization
                        fig = px.histogram(
                            results_df, 
                            x='prediction', 
                            title='Spam vs Ham Distribution',
                            color='prediction',
                            color_discrete_map={'spam': 'red', 'ham': 'green'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Results table
                        st.subheader("Detailed Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name='spam_detection_results.csv',
                            mime='text/csv'
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        # Manual batch input
        st.subheader("Manual Batch Input")
        batch_text = st.text_area(
            "Enter multiple messages (one per line):",
            height=200,
            placeholder="Message 1\nMessage 2\nMessage 3\n..."
        )
        
        if st.button("🔍 Analyze Batch", key="manual_batch"):
            if batch_text.strip():
                messages = [msg.strip() for msg in batch_text.split('\n') if msg.strip()]
                
                if messages:
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, message in enumerate(messages):
                        prediction, confidence, probabilities = detector.predict(message)
                        results.append({
                            'message': message,
                            'prediction': prediction,
                            'confidence': confidence,
                            'spam_probability': probabilities[1]
                        })
                        progress_bar.progress((i + 1) / len(messages))
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                else:
                    st.warning("Please enter at least one message.")
    
    with tab3:
        st.header("Sample Test Messages")
        st.markdown("Test the detector with these sample messages")
        
        samples = get_sample_messages()
        
        for category, messages in samples.items():
            st.subheader(f"{category.title()} Messages")
            
            for i, message in enumerate(messages):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.text_area(
                        f"Sample {i+1}:",
                        value=message,
                        height=60,
                        key=f"{category}_{i}_text",
                        disabled=True
                    )
                
                with col2:
                    if st.button(f"Test", key=f"{category}_{i}_test"):
                        prediction, confidence, probabilities = detector.predict(message)
                        
                        if prediction == 'spam':
                            st.error("🚨 SPAM")
                        else:
                            st.success("✅ NOT SPAM")
                        st.write(f"Confidence: {confidence:.2%}")
    
    with tab4:
        st.header("Model Information")
        
        # Model performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Statistics")
            stats = detector.get_model_stats()
            for key, value in stats.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("🔑 Top Spam Features")
            spam_features = detector.get_top_spam_features()
            for i, feature in enumerate(spam_features[:15], 1):
                st.write(f"{i}. {feature}")
        
        # Feature distribution
        st.subheader("📈 Feature Importance")
        features = detector.get_feature_importance()
        if features:
            fig = px.bar(
                x=list(features.values())[:20],
                y=list(features.keys())[:20],
                orientation='h',
                title='Top 20 Most Important Features',
                labels={'x': 'Importance Score', 'y': 'Features'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
