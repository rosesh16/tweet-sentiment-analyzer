# Dataset Guide for Sentiment Analysis

This guide explains how to use real datasets instead of sample data for training your sentiment model.

## Supported Formats

### 1. CSV Files (Recommended)
Your CSV should have at least two columns:
- **Text column**: `text`, `tweet`, `message`, or `content`
- **Label column**: `sentiment`, `label`, or `airline_sentiment`

#### Example CSV format:
```csv
text,sentiment
"I love this product!",positive
"This is terrible",negative
"Amazing experience",positive
```

or with numeric labels:
```csv
tweet,label
"I love this product!",1
"This is terrible",0
"Amazing experience",1
```

## Popular Datasets

### 1. **Sentiment140** (1.6M tweets)
- **Download**: https://www.kaggle.com/datasets/kazanova/sentiment140
- **Format**: CSV with columns: target, ids, date, flag, user, text
- **Labels**: 0 = negative, 4 = positive
- **Size**: ~238 MB

### 2. **Twitter US Airline Sentiment** (14k tweets)
- **Download**: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
- **Format**: CSV with airline_sentiment (positive/negative/neutral)
- **Size**: ~3 MB
- **Best for**: Quick training and testing

### 3. **Twitter Sentiment Analysis** (1.6M tweets)
- **Download**: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
- **Format**: CSV with Sentiment (Positive/Negative/Neutral)
- **Size**: Medium

## How to Use Your Dataset

### Method 1: Interactive (Recommended)
Simply run the training script and provide the path when prompted:

```bash
python train_model.py
```

Then enter the path to your CSV:
```
Enter path to your CSV file: C:\path\to\your\dataset.csv
```

### Method 2: Direct Path
Edit the script at the bottom:
```python
train_model_with_dataset('path/to/your/dataset.csv')
```

### Method 3: Default Location
Place your CSV file as `twitter_sentiment.csv` in the project root, and the script will auto-detect it.

## Dataset Requirements

### Minimum Requirements:
- At least 1,000 samples (more is better)
- Balanced classes (similar number of positive/negative)
- Clean text (no excessive noise)

### Recommended:
- 10,000+ samples for good accuracy
- 80/20 train-test split (handled automatically)
- Pre-processed text (lowercase, no URLs)

## SQL Database Support

To use data from SQL database, first export to CSV:

### PostgreSQL:
```sql
COPY (SELECT text, sentiment FROM tweets) 
TO 'C:\path\to\dataset.csv' 
WITH CSV HEADER;
```

### MySQL:
```sql
SELECT text, sentiment 
INTO OUTFILE 'C:\path\to\dataset.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
FROM tweets;
```

### SQLite:
```bash
sqlite3 database.db
.headers on
.mode csv
.output dataset.csv
SELECT text, sentiment FROM tweets;
.quit
```

### Python (pandas):
```python
import pandas as pd
import sqlite3

# For SQLite
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT text, sentiment FROM tweets", conn)
df.to_csv('dataset.csv', index=False)

# For PostgreSQL/MySQL
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@localhost/dbname')
df = pd.read_sql_query("SELECT text, sentiment FROM tweets", engine)
df.to_csv('dataset.csv', index=False)
```

## Creating Your Own Dataset

If you want to create a custom dataset:

```python
import pandas as pd

data = {
    'text': [
        'Your tweet text here',
        'Another tweet',
        # ... more tweets
    ],
    'sentiment': [
        'positive',
        'negative',
        # ... corresponding labels
    ]
}

df = pd.DataFrame(data)
df.to_csv('my_dataset.csv', index=False)
```

## Expected Output

When training with a real dataset, you'll see:

```
============================================================
Twitter Sentiment Analysis - Model Training
============================================================

Loading dataset from twitter_sentiment.csv...
Dataset loaded: 14640 rows
Columns found: ['text', 'sentiment', 'airline']

Using text column: 'text', label column: 'sentiment'
Processed 14640 samples
Positive samples: 5462
Negative samples: 9178

Splitting data into train/test sets...
Training samples: 11712
Testing samples: 2928

Training model...

Evaluating model...
Accuracy: 89.23%

Classification Report:
              precision    recall  f1-score   support

    Negative       0.91      0.94      0.92      1836
    Positive       0.86      0.81      0.83      1092

✅ Model saved as 'sentiment_model.pkl'
```

## Troubleshooting

### "Could not identify text and label columns"
- Ensure your CSV has columns named: `text`/`tweet` and `sentiment`/`label`
- Or rename your columns to match

### "File not found"
- Use absolute path: `C:\Users\...\dataset.csv`
- Or place file in project root

### Low Accuracy
- Use more training data (10k+ samples)
- Ensure balanced classes
- Clean your data (remove spam, duplicates)
- Try different models (Logistic Regression vs Naive Bayes)

## Best Practices

1. **Start Small**: Test with 10k samples first
2. **Validate**: Check accuracy on test set
3. **Iterate**: Try different preprocessing/models
4. **Version Control**: Save different model versions
5. **Document**: Note which dataset achieved what accuracy
