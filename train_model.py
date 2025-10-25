from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Sample training data (positive=1, negative=0)
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

# Create pipeline with TF-IDF and Naive Bayes
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', MultinomialNB())
])

# Train the model
print("Training sentiment model...")
pipe.fit(train_texts, train_labels)

# Save the model
joblib.dump(pipe, 'sentiment_model.pkl')
print("Model saved as 'sentiment_model.pkl'")

# Test the model
test_samples = [
    "I love this so much!",
    "This is terrible and bad"
]

for text in test_samples:
    pred = pipe.predict([text])[0]
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"'{text}' -> {sentiment}")
