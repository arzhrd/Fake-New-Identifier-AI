import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import warnings
warnings.filterwarnings('ignore')

class FakeNewsPredictor:
    def __init__(self, model_path='fake_news_model.pkl'):
        """Initialize the predictor with saved model"""
        try:
            # Download NLTK data if not already downloaded
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            
            # Load the saved model components
            with open(model_path, 'rb') as file:
                model_data = pickle.load(file)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.model_name = model_data['model_name']
            
            print(f"Loaded {self.model_name} model successfully!")
            
        except FileNotFoundError:
            print(f"Error: Model file '{model_path}' not found!")
            print("Please run 'train_model.py' first to create the model.")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        
        # Stemming
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        
        return ' '.join(stemmed_words)
    
    def predict_single_news(self, text):
        """Predict if a single news article is fake or real"""
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            
            if not processed_text.strip():
                return {
                    'error': 'Empty or invalid text provided',
                    'prediction': None,
                    'confidence': None
                }
            
            # Convert to TF-IDF features
            text_tfidf = self.vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = self.model.predict(text_tfidf)[0]
            probabilities = self.model.predict_proba(text_tfidf)[0]
            
            # Prepare result
            result = {
                'original_text': text,
                'prediction': 'Fake' if prediction == 1 else 'Real',
                'confidence': max(probabilities),
                'probabilities': {
                    'Real': probabilities[0],
                    'Fake': probabilities[1]
                },
                'model_used': self.model_name
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f'Prediction error: {e}',
                'prediction': None,
                'confidence': None
            }
    
    def predict_batch_news(self, news_list):
        """Predict multiple news articles at once"""
        results = []
        
        for i, news_text in enumerate(news_list):
            print(f"Processing article {i+1}/{len(news_list)}...")
            result = self.predict_single_news(news_text)
            results.append(result)
        
        return results
    
    def predict_from_file(self, file_path, text_column='text'):
        """Predict news from a CSV file"""
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            if text_column not in df.columns:
                return f"Error: Column '{text_column}' not found in the file."
            
            # Get predictions for all texts
            predictions = []
            confidences = []
            fake_probs = []
            
            print(f"Processing {len(df)} articles from file...")
            
            for idx, text in enumerate(df[text_column]):
                if idx % 50 == 0:  # Progress indicator
                    print(f"Processed {idx}/{len(df)} articles...")
                
                result = self.predict_single_news(text)
                
                if result.get('error'):
                    predictions.append('Error')
                    confidences.append(0)
                    fake_probs.append(0)
                else:
                    predictions.append(result['prediction'])
                    confidences.append(result['confidence'])
                    fake_probs.append(result['probabilities']['Fake'])
            
            # Add results to dataframe
            df['predicted_label'] = predictions
            df['confidence'] = confidences
            df['fake_probability'] = fake_probs
            
