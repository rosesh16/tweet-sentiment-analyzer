import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Block 2: Load Dataset
print("=== Loading Dataset ===")
CSV_FILE = 'kaggle_tweets.csv'
df = pd.read_csv(CSV_FILE, encoding='latin-1', header=None, 
                 names=['sentiment', 'id', 'date', 'query', 'user', 'text'])

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Block 3: Data Preprocessing
print("\n=== Data Preprocessing ===")
text_col = 'text'
label_col = 'sentiment'

print(f"Using text column: '{text_col}'")
print(f"Using label column: '{label_col}'")

# Extract data
texts = df[text_col].astype(str).tolist()
labels = df[label_col].tolist()

# Convert labels to binary (0=negative, 1=positive)
if isinstance(labels[0], str):
    label_map = {
        'positive': 1, 'pos': 1, '1': 1, 1: 1,
        'negative': 0, 'neg': 0, '0': 0, 0: 0,
        'neutral': 1  # Treat neutral as positive
    }
    labels = [label_map.get(str(l).lower(), 0) for l in labels]
else:
    # Sentiment140 format: 0=negative, 4=positive
    labels = [1 if l > 2 else 0 for l in labels]

print(f"\nProcessed {len(texts)} samples")
print(f"Positive: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
print(f"Negative: {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

# Block 4: Sample the Data (Optional)
print("\n=== Sampling Data ===")
MAX_SAMPLES = 100000  # Set to None to use all data

if MAX_SAMPLES and len(texts) > MAX_SAMPLES:
    print(f"Using {MAX_SAMPLES:,} samples out of {len(texts):,}")
    # Use stratified sampling to preserve class distribution
    from sklearn.model_selection import train_test_split
    texts_sample, _, labels_sample, _ = train_test_split(
        texts, labels, train_size=MAX_SAMPLES, random_state=42, stratify=labels
    )
    texts = texts_sample
    labels = labels_sample
    print(f"Positive after sampling: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"Negative after sampling: {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
else:
    print(f"Using all {len(texts):,} samples")

# Block 5: Train/Test Split
print("\n=== Train/Test Split ===")
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples: {len(X_test):,}")

# Block 6: Model Training - Logistic Regression
print("\n=== Training Logistic Regression ===")
model_lr = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Train
model_lr.fit(X_train, y_train)

# Predict
y_pred_lr = model_lr.predict(X_test)

# Evaluate
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f"✅ Logistic Regression Accuracy: {accuracy_lr:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Negative', 'Positive']))

# Block 7: Model Training - Naive Bayes
print("\n=== Training Naive Bayes ===")
model_nb = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )),
    ('clf', MultinomialNB())
])

model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

print(f"✅ Naive Bayes Accuracy: {accuracy_nb:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb, target_names=['Negative', 'Positive']))

# Best Model Comparison
print(f"\n🏆 Best Model: {'Logistic Regression' if accuracy_lr > accuracy_nb else 'Naive Bayes'}")
print(f"Best Accuracy: {max(accuracy_lr, accuracy_nb):.2%}")

# Block 8: Test with Custom Examples
print("\n=== Testing Custom Examples ===")
best_model = model_lr if accuracy_lr > accuracy_nb else model_nb

test_tweets = [
    "I absolutely love this product! Best purchase ever!",
    "This is terrible. Worst experience of my life.",
    "Having an amazing day! Everything is going great!",
    "Very disappointed and frustrated with this service.",
    "Not bad, could be better though."
]

for tweet in test_tweets:
    pred = best_model.predict([tweet])[0]
    sentiment = "Positive 😊" if pred == 1 else "Negative 😢"
    print(f"'{tweet}'")
    print(f"  → {sentiment}\n")

# Block 9: Save the Best Model
print("=== Saving Model ===")
MODEL_FILE = 'sentiment_model.pkl'
best_model = model_lr if accuracy_lr > accuracy_nb else model_nb

joblib.dump(best_model, MODEL_FILE)
print(f"✅ Model saved as '{MODEL_FILE}'")
print(f"   Accuracy: {max(accuracy_lr, accuracy_nb):.2%}")