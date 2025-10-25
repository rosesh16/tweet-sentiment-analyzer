# 🐦 Tweet Sentiment Analyzer

A real-time sentiment analysis application that classifies tweets as positive or negative using Machine Learning.

## Features

- 🎯 **Instant Analysis** - Get sentiment predictions in real-time
- 🎨 **Beautiful UI** - Modern, Twitter-inspired interface
- 📊 **Machine Learning** - Powered by scikit-learn's ML algorithms
- 💡 **Example Tweets** - Quick-test with pre-loaded examples
- 📱 **Responsive** - Works on desktop and mobile

## Tech Stack

- **Frontend**: Streamlit
- **ML Model**: scikit-learn (TF-IDF + Naive Bayes)
- **Language**: Python 3.13

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd AI-AGENT
```

2. Create virtual environment:
```bash
python -m venv .venv
```

3. Activate virtual environment:
- Windows: `.\.venv\Scripts\Activate.ps1`
- Mac/Linux: `source .venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Train the model:
```bash
python train_model.py
```

6. Run the app:
```bash
streamlit run app.py
```

## Usage

1. Open the app in your browser (usually http://localhost:8501)
2. Enter a tweet or select an example
3. Click "Analyze Sentiment"
4. See the prediction with visual feedback!

## Deployment

This app can be deployed on:
- **Streamlit Cloud** (Recommended - Free)
- **Heroku**
- **AWS/GCP/Azure**

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy!

## Project Structure

```
AI-AGENT/
├── app.py                  # Main Streamlit application
├── train_model.py          # Model training script
├── sentiment_model.pkl     # Trained model file
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit configuration
└── README.md              # This file
```

## Model Details

- **Algorithm**: Naive Bayes Classifier
- **Features**: TF-IDF (1-2 grams, max 5000 features)
- **Training**: Basic positive/negative tweet dataset

## Future Enhancements

- [ ] Add neutral sentiment category
- [ ] Display confidence scores
- [ ] Add sentiment history tracking
- [ ] Integrate with Twitter API
- [ ] Multi-language support
- [ ] Advanced visualizations

## License

MIT License

## Author

Built with ❤️ using Streamlit & scikit-learn
