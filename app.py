from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import plotly.graph_objs as go
import plotly.utils
import json
from werkzeug.utils import secure_filename
import warnings
from datetime import datetime
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Ensure required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models/', exist_ok=True)
os.makedirs('exports/', exist_ok=True)

# Load the trained model
print("🔍 Loading model...")
try:
    model = joblib.load('models/best_random_forest_sentiment_model.joblib')
    print("✅ Model loaded successfully!")
    MODEL_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Warning: Could not load model - {str(e)}")
    print("📝 Falling back to TextBlob for sentiment analysis")
    model = None
    MODEL_AVAILABLE = False

class PriceOptimizer:
    def __init__(self, model=None):
        self.model = model
        self.use_ml_model = model is not None
        logger.info(f"PriceOptimizer initialized (ML Model: {self.use_ml_model})")
    
    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob (fallback method)"""
        try:
            if pd.isna(text) or str(text).strip() == '':
                return 'neutral', 0.0
            
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'positive', polarity
            elif polarity < -0.1:
                return 'negative', polarity
            else:
                return 'neutral', polarity
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {str(e)}")
            return 'neutral', 0.0
    
    def analyze_sentiment_ml(self, text):
        """Analyze sentiment using trained ML model"""
        try:
            if pd.isna(text) or str(text).strip() == '':
                return 'neutral', 0.0
            
            # Predict using the model
            # Note: Assumes model is a pipeline with vectorizer
            prediction = self.model.predict([str(text)])[0]
            
            # Get prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba([str(text)])[0]
                confidence = max(probabilities)
            else:
                confidence = 0.7  # Default confidence
            
            # Map prediction to sentiment (adjust based on your model's output)
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment = sentiment_map.get(prediction, 'neutral')
            
            # Calculate polarity score from prediction
            if sentiment == 'positive':
                polarity = confidence * 0.5  # 0 to 0.5
            elif sentiment == 'negative':
                polarity = -confidence * 0.5  # -0.5 to 0
            else:
                polarity = 0.0
            
            return sentiment, polarity
            
        except Exception as e:
            logger.error(f"Error with ML sentiment analysis: {str(e)}")
            # Fallback to TextBlob
            return self.analyze_sentiment_textblob(text)
    
    def analyze_sentiment(self, text):
        """Main sentiment analysis method - uses ML model or TextBlob"""
        if self.use_ml_model:
            return self.analyze_sentiment_ml(text)
        else:
            return self.analyze_sentiment_textblob(text)
    
    def extract_features(self, df):
        """Extract features from the dataset for analysis"""
        logger.info("Extracting features from dataset")
        features_df = df.copy()
        
        # Clean the data
        features_df['review'] = features_df['review'].fillna('')
        features_df['product_name'] = features_df['product_name'].fillna('Unknown Product')
        features_df['price'] = pd.to_numeric(features_df['price'], errors='coerce')
        
        # Remove rows with invalid prices
        initial_count = len(features_df)
        features_df = features_df.dropna(subset=['price'])
        features_df = features_df[features_df['price'] > 0]
        removed_count = initial_count - len(features_df)
        
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} rows with invalid prices")
        
        # Analyze sentiment for each review
        sentiments = []
        sentiment_scores = []
        
        logger.info("Analyzing sentiment for reviews...")
        for idx, review in enumerate(features_df['review']):
            if idx % 100 == 0 and idx > 0:
                logger.info(f"Processed {idx}/{len(features_df)} reviews")
            
            sentiment, score = self.analyze_sentiment(review)
            sentiments.append(sentiment)
            sentiment_scores.append(score)
        
        features_df['sentiment'] = sentiments
        features_df['sentiment_score'] = sentiment_scores
        
        # Create additional features
        features_df['review_length'] = features_df['review'].astype(str).str.len()
        features_df['word_count'] = features_df['review'].astype(str).str.split().str.len()
        
        # Encode categorical sentiment
        sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        features_df['sentiment_encoded'] = features_df['sentiment'].map(sentiment_mapping)
        
        logger.info(f"Feature extraction completed. Dataset shape: {features_df.shape}")
        return features_df
    
    def optimize_price(self, df):
        """Optimize prices based on sentiment analysis and advanced pricing strategies"""
        logger.info("Starting price optimization")
        feature_df = self.extract_features(df)
        
        if len(feature_df) == 0:
            raise ValueError("No valid data remaining after cleaning")
        
        optimized_df = feature_df.copy()
        
        # Advanced price optimization logic
        def calculate_optimized_price(row):
            original_price = row['price']
            sentiment = row['sentiment']
            sentiment_score = row['sentiment_score']
            review_length = row['review_length']
            word_count = row['word_count']
            
            # Base multiplier based on sentiment
            if sentiment == 'positive':
                # Increase price for positive sentiment (5% to 20% based on sentiment strength)
                base_multiplier = 1 + (abs(sentiment_score) * 0.20)
                
                # Additional boost for detailed positive reviews
                if review_length > 100:
                    base_multiplier += 0.05
                if word_count > 20:
                    base_multiplier += 0.03
                    
            elif sentiment == 'negative':
                # Decrease price for negative sentiment (5% to 25% based on sentiment strength)
                base_multiplier = 1 + (sentiment_score * 0.25)  # sentiment_score is negative
                
                # Additional reduction for detailed negative reviews
                if review_length > 100:
                    base_multiplier -= 0.05
                if word_count > 20:
                    base_multiplier -= 0.03
                    
            else:  # neutral
                # Small adjustment for neutral sentiment
                base_multiplier = 1.02
                
                # Slight boost for detailed neutral reviews (shows engagement)
                if review_length > 150:
                    base_multiplier += 0.02
            
            # Ensure reasonable bounds
            base_multiplier = max(0.7, min(1.3, base_multiplier))
            
            optimized_price = original_price * base_multiplier
            return round(optimized_price, 2)
        
        optimized_df['optimized_price'] = optimized_df.apply(calculate_optimized_price, axis=1)
        optimized_df['price_change'] = optimized_df['optimized_price'] - optimized_df['price']
        optimized_df['price_change_percent'] = (optimized_df['price_change'] / optimized_df['price']) * 100
        
        # Calculate confidence score based on review quality
        optimized_df['confidence_score'] = self.calculate_confidence_score(optimized_df)
        
        logger.info("Price optimization completed")
        return optimized_df
    
    def calculate_confidence_score(self, df):
        """Calculate confidence score for price optimization"""
        confidence_scores = []
        
        for _, row in df.iterrows():
            score = 0.5  # Base confidence
            
            # Increase confidence for longer, more detailed reviews
            if row['review_length'] > 50:
                score += 0.2
            if row['review_length'] > 100:
                score += 0.1
            
            # Increase confidence for strong sentiment (positive or negative)
            sentiment_strength = abs(row['sentiment_score'])
            score += sentiment_strength * 0.3
            
            # Cap at 1.0
            score = min(1.0, score)
            confidence_scores.append(round(score, 3))
        
        return confidence_scores
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations for price optimization analysis"""
        logger.info("Creating visualizations")
        plots = {}
        
        try:
            # 1. Sentiment Distribution
            sentiment_counts = df['sentiment'].value_counts()
            colors_map = {'negative': '#ff4757', 'neutral': '#ffa502', 'positive': '#2ed573'}
            colors = [colors_map.get(sent, '#747d8c') for sent in sentiment_counts.index]
            
            fig_sentiment = go.Figure(data=[
                go.Bar(
                    x=sentiment_counts.index, 
                    y=sentiment_counts.values,
                    marker_color=colors,
                    text=sentiment_counts.values,
                    textposition='auto'
                )
            ])
            fig_sentiment.update_layout(
                title='Sentiment Distribution in Reviews',
                xaxis_title='Sentiment',
                yaxis_title='Count',
                showlegend=False,
                height=400
            )
            plots['sentiment_dist'] = json.dumps(fig_sentiment, cls=plotly.utils.PlotlyJSONEncoder)
            
            # 2. Price Distribution (Before vs After)
            fig_price = go.Figure()
            fig_price.add_trace(go.Histogram(
                x=df['price'], 
                name='Original Price', 
                opacity=0.7,
                marker_color='#3742fa',
                nbinsx=20
            ))
            fig_price.add_trace(go.Histogram(
                x=df['optimized_price'], 
                name='Optimized Price', 
                opacity=0.7,
                marker_color='#2ed573',
                nbinsx=20
            ))
            fig_price.update_layout(
                title='Price Distribution: Before vs After Optimization',
                xaxis_title='Price ($)',
                yaxis_title='Frequency',
                barmode='overlay',
                height=400
            )
            plots['price_dist'] = json.dumps(fig_price, cls=plotly.utils.PlotlyJSONEncoder)
            
            # 3. Price Change by Sentiment
            fig_change = go.Figure()
            colors_map = {'positive': '#2ed573', 'neutral': '#ffa502', 'negative': '#ff4757'}
            
            for sentiment in df['sentiment'].unique():
                sentiment_data = df[df['sentiment'] == sentiment]
                fig_change.add_trace(go.Box(
                    y=sentiment_data['price_change_percent'],
                    name=sentiment.title(),
                    marker_color=colors_map.get(sentiment, '#747d8c'),
                    boxpoints='outliers'
                ))
            fig_change.update_layout(
                title='Price Change Percentage by Sentiment',
                xaxis_title='Sentiment',
                yaxis_title='Price Change (%)',
                height=400
            )
            plots['price_change'] = json.dumps(fig_change, cls=plotly.utils.PlotlyJSONEncoder)
            
            # 4. Sentiment Score vs Original Price Scatter
            color_map = {'positive': '#2ed573', 'neutral': '#ffa502', 'negative': '#ff4757'}
            fig_scatter = go.Figure()
            
            for sentiment in df['sentiment'].unique():
                sentiment_data = df[df['sentiment'] == sentiment]
                fig_scatter.add_trace(go.Scatter(
                    x=sentiment_data['price'],
                    y=sentiment_data['sentiment_score'],
                    mode='markers',
                    name=sentiment.title(),
                    marker=dict(
                        color=color_map[sentiment], 
                        size=8,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Price: $%{x}<br>' +
                                  'Sentiment Score: %{y}<br>' +
                                  '<extra></extra>',
                    text=sentiment_data['product_name']
                ))
            
            fig_scatter.update_layout(
                title='Sentiment Score vs Original Price',
                xaxis_title='Original Price ($)',
                yaxis_title='Sentiment Score',
                height=400
            )
            plots['sentiment_price'] = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)
            
            logger.info("Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            plots['error'] = str(e)
        
        return plots

# Initialize price optimizer
price_optimizer = PriceOptimizer(model)

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_AVAILABLE,
        'sentiment_method': 'ML Model' if MODEL_AVAILABLE else 'TextBlob',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process price optimization"""
    logger.info("File upload request received")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400
    
    try:
        # Read the uploaded CSV
        df = pd.read_csv(file)
        logger.info(f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns")
        
        # Validate required columns
        required_columns = ['product_name', 'price', 'review']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error_msg = f'Missing required columns: {", ".join(missing_columns)}. Required columns are: product_name, price, review'
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
        
        if len(df) == 0:
            return jsonify({'error': 'The uploaded CSV file is empty'}), 400
        
        # Perform price optimization
        logger.info("Starting price optimization process")
        optimized_df = price_optimizer.optimize_price(df)
        
        # Create visualizations
        plots = price_optimizer.create_visualizations(optimized_df)
        
        # Calculate summary statistics
        summary_stats = {
            'total_products': len(optimized_df),
            'avg_original_price': round(optimized_df['price'].mean(), 2),
            'avg_optimized_price': round(optimized_df['optimized_price'].mean(), 2),
            'avg_price_change': round(optimized_df['price_change'].mean(), 2),
            'avg_price_change_percent': round(optimized_df['price_change_percent'].mean(), 2),
            'positive_reviews': len(optimized_df[optimized_df['sentiment'] == 'positive']),
            'neutral_reviews': len(optimized_df[optimized_df['sentiment'] == 'neutral']),
            'negative_reviews': len(optimized_df[optimized_df['sentiment'] == 'negative']),
            'total_revenue_change': round((optimized_df['optimized_price'].sum() - optimized_df['price'].sum()), 2),
            'avg_confidence': round(optimized_df['confidence_score'].mean(), 3),
            'high_confidence_products': len(optimized_df[optimized_df['confidence_score'] > 0.8]),
            'sentiment_method': 'ML Model' if MODEL_AVAILABLE else 'TextBlob'
        }
        
        # Convert dataframe to JSON for display
        display_columns = ['product_name', 'price', 'optimized_price', 'sentiment', 
                          'price_change', 'price_change_percent', 'confidence_score']
        # Show all results, not just 50
        results_data = optimized_df[display_columns].to_dict('records')
        
        # Store results for download functionality
        optimized_df.to_csv('exports/latest_optimization.csv', index=False)
        
        logger.info("Price optimization completed successfully")
        
        return jsonify({
            'success': True,
            'summary': summary_stats,
            'plots': plots,
            'data': results_data,
            'message': f'Successfully optimized prices for {len(optimized_df)} products!'
        })
        
    except pd.errors.EmptyDataError:
        error_msg = 'The uploaded file is empty or corrupted'
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 400
    
    except pd.errors.ParserError:
        error_msg = 'Unable to parse CSV file. Please check the file format'
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 400
    
    except ValueError as e:
        error_msg = str(e)
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 400
    
    except Exception as e:
        error_msg = f'Error processing file: {str(e)}'
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/download/csv')
def download_csv():
    """Download the latest optimization results as CSV"""
    try:
        file_path = 'exports/latest_optimization.csv'
        if os.path.exists(file_path):
            return send_file(
                file_path,
                as_attachment=True,
                download_name=f'price_optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mimetype='text/csv'
            )
        else:
            return jsonify({'error': 'No optimization results available for download'}), 404
    except Exception as e:
        logger.error(f"Error downloading CSV: {str(e)}")
        return jsonify({'error': 'Error generating download'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size allowed is 16MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error occurred'}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle not found errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

if __name__ == '__main__':
    print("🚀 Starting Price Optimization Dashboard...")
    print(f"📊 Model Status: {'✅ ML Model Loaded' if MODEL_AVAILABLE else '⚠️ Using TextBlob (ML model not available)'}")
    print("🌐 Access the application at: http://localhost:5000")
    print("📁 Upload a CSV with columns: product_name, price, review")
    print("-" * 50)
    
    app.run(debug=True, port=5000, host='0.0.0.0')