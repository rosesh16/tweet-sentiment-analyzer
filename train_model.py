from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
import os
import urllib.request
import zipfile

def download_twitter_dataset():
    """
    Downloads the Sentiment140 dataset (1.6M tweets)
    Alternative: Twitter US Airline Sentiment dataset
    """
    print("Checking for dataset...")
    
    # Option 1: Use a smaller, pre-processed dataset
    dataset_url = "https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv"
    dataset_file = "twitter_sentiment.csv"
    
    if not os.path.exists(dataset_file):
        print(f"Downloading dataset from {dataset_url}...")
        try:
            urllib.request.urlretrieve(dataset_url, dataset_file)
            print("Dataset downloaded successfully!")
        except Exception as e:
            print(f"Could not download dataset: {e}")
            print("\nPlease download a dataset manually:")
            print("1. Sentiment140: https://www.kaggle.com/datasets/kazanova/sentiment140")
            print("2. Twitter Airline: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment")
            print("3. Or provide your own CSV with 'text' and 'sentiment' columns")
            return None
    else:
        print(f"Found existing dataset: {dataset_file}")
    
    return dataset_file

def load_dataset(filepath=None):
    """
    Load dataset from CSV file.
    Expected columns: 'text' and 'sentiment' (0=negative, 1=positive)
    """
    if filepath is None:
        filepath = "twitter_sentiment.csv"
    
    print(f"Loading dataset from {filepath}...")
    
    try:
        # Try different common formats
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, encoding='latin-1')
            print(f"Dataset loaded: {len(df)} rows")
            print(f"Columns found: {df.columns.tolist()}")
            return df
        else:
            print(f"File not found: {filepath}")
            return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def prepare_data(df, text_column='text', label_column='sentiment'):
    """
    Prepare data for training.
    Handles different dataset formats.
    """
    # Try to identify the text and label columns
    possible_text_cols = ['text', 'tweet', 'message', 'content', 'Tweet', 'TweetText']
    possible_label_cols = ['sentiment', 'label', 'Sentiment', 'airline_sentiment']
    
    # Find the actual column names
    text_col = None
    label_col = None
    
    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break
    
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break
    
    if text_col is None or label_col is None:
        print(f"Could not identify text and label columns.")
        print(f"Available columns: {df.columns.tolist()}")
        print("\nPlease ensure your CSV has columns named 'text' and 'sentiment'")
        return None, None
    
    print(f"Using text column: '{text_col}', label column: '{label_col}'")
    
    # Extract texts and labels
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].tolist()
    
    # Convert labels to binary (0=negative, 1=positive)
    # Handle different label formats
    if isinstance(labels[0], str):
        # String labels like 'positive', 'negative'
        label_map = {
            'positive': 1, 'pos': 1, '1': 1, 1: 1,
            'negative': 0, 'neg': 0, '0': 0, 0: 0,
            'neutral': 1  # Treat neutral as positive for binary classification
        }
        labels = [label_map.get(str(l).lower(), 0) for l in labels]
    else:
        # Numeric labels - assume 4 or 5 = positive, 0-2 = negative (Sentiment140 format)
        labels = [1 if l > 2 else 0 for l in labels]
    
    print(f"Processed {len(texts)} samples")
    print(f"Positive samples: {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")
    
    return texts, labels

def train_model_with_dataset(csv_file=None):
    """
    Train model using a CSV dataset
    """
    # Load dataset
    if csv_file is None:
        csv_file = download_twitter_dataset()
    
    if csv_file is None or not os.path.exists(csv_file):
        print("\n⚠️ No dataset found. Creating a basic model with sample data...")
        return train_basic_model()
    
    df = load_dataset(csv_file)
    if df is None:
        print("\n⚠️ Could not load dataset. Creating a basic model with sample data...")
        return train_basic_model()
    
    # Prepare data
    texts, labels = prepare_data(df)
    if texts is None:
        print("\n⚠️ Could not prepare data. Creating a basic model with sample data...")
        return train_basic_model()
    
    # Limit dataset size for faster training (optional)
    max_samples = 100000  # Use up to 100k samples
    if len(texts) > max_samples:
        print(f"Using {max_samples} samples for training (dataset has {len(texts)} total)")
        texts = texts[:max_samples]
        labels = labels[:max_samples]
    
    # Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create pipeline with TF-IDF and Logistic Regression (better than Naive Bayes)
    print("\nTraining model...")
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    pipe.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Save the model
    joblib.dump(pipe, 'sentiment_model.pkl')
    print("\n✅ Model saved as 'sentiment_model.pkl'")
    
    # Test the model
    test_samples = [
        "I absolutely love this! Best experience ever!",
        "This is terrible and disappointing",
        "Having an amazing day today!",
        "Worst service I've ever encountered"
    ]
    
    print("\nTesting model with sample tweets:")
    for text in test_samples:
        pred = pipe.predict([text])[0]
        sentiment = "Positive 😊" if pred == 1 else "Negative 😢"
        print(f"  '{text}' → {sentiment}")
    
    return pipe

def train_basic_model():
    """
    Fallback: Train with basic sample data if dataset is not available
    """
    print("\nTraining basic model with sample data...")
    
    train_texts = [
        # Positive tweets
        "I love this product! It's amazing!",
        "Best day ever! So happy right now",
        "This is absolutely fantastic!",
        "Great experience, highly recommend",
        "Wonderful service and quality",
        "I'm so excited about this!",
        "Perfect! Exactly what I needed",
        "This made my day! Thank you",
        "Awesome work, keep it up!",
        "Incredible results, very satisfied",
        
        # Negative tweets
        "This is terrible, I hate it",
        "Worst experience ever",
        "Absolutely disappointed",
        "Not worth the money, poor quality",
        "I'm so frustrated with this",
        "Horrible customer service",
        "This is a complete waste of time",
        "Very unhappy with this purchase",
        "Terrible product, do not buy",
        "Disappointing and frustrating"
    ]
    
    train_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', MultinomialNB())
    ])
    
    pipe.fit(train_texts, train_labels)
    
    joblib.dump(pipe, 'sentiment_model.pkl')
    print("✅ Basic model saved as 'sentiment_model.pkl'")
    
    return pipe

if __name__ == "__main__":
    print("="*60)
    print("Twitter Sentiment Analysis - Model Training")
    print("="*60)
    
    # Check if user has a custom dataset
    custom_csv = input("\nEnter path to your CSV file (or press Enter to use default): ").strip()
    
    if custom_csv and os.path.exists(custom_csv):
        print(f"\nUsing custom dataset: {custom_csv}")
        train_model_with_dataset(custom_csv)
    else:
        train_model_with_dataset()
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
