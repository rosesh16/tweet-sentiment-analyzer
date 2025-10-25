# 🚀 Sentiment Analysis App - Google Cloud Deployment Guide

## 📋 Prerequisites

Before deploying, ensure you have:

1. **Google Cloud Account** with billing enabled
2. **Google Cloud CLI** installed on your machine
3. **Docker** installed (optional, for local testing)
4. **Python 3.10+** installed locally

## 🛠️ Setup Instructions

### Step 1: Install Google Cloud CLI

**Windows:**
```powershell
# Download and install from: https://cloud.google.com/sdk/docs/install
# Or use Chocolatey:
choco install gcloudsdk
```

**macOS:**
```bash
# Using Homebrew
brew install google-cloud-sdk
```

**Linux:**
```bash
# Follow instructions at: https://cloud.google.com/sdk/docs/install
curl https://sdk.cloud.google.com | bash
```

### Step 2: Authenticate and Setup Project

```bash
# Login to your Google Cloud account
gcloud auth login

# Create a new project (replace YOUR_PROJECT_ID with unique ID)
gcloud projects create YOUR_PROJECT_ID --name="Sentiment Analysis App"

# Set the project as default
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable appengine.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### Step 3: Test Locally (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Test the Flask app
python main.py
```

Visit `http://localhost:8080` to test the application.

## 🚀 Deployment Options

### Option A: Google App Engine (Recommended)

**Advantages:**
- Fully managed
- Auto-scaling
- Easy deployment
- Built-in load balancing

**Steps:**

1. **Initialize App Engine:**
```bash
gcloud app create --region=us-central1
```

2. **Deploy the application:**
```bash
gcloud app deploy app.yaml
```

3. **View your app:**
```bash
gcloud app browse
```

**Expected Output:**
```
Deployed service [default] to [https://YOUR_PROJECT_ID.uc.r.appspot.com]
```

### Option B: Google Cloud Run

**Advantages:**
- Container-based
- Pay-per-request
- Better for variable traffic

**Steps:**

1. **Build and push Docker image:**
```bash
# Set up Docker authentication
gcloud auth configure-docker

# Build the image
docker build -t gcr.io/YOUR_PROJECT_ID/sentiment-app .

# Push to Container Registry
docker push gcr.io/YOUR_PROJECT_ID/sentiment-app
```

2. **Deploy to Cloud Run:**
```bash
gcloud run deploy sentiment-app \
    --image gcr.io/YOUR_PROJECT_ID/sentiment-app \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1
```

## 📁 Project Structure

```
sentiment-analysis-app/
├── main.py                 # Flask application
├── sentiment_model.pkl     # Trained ML model
├── templates/
│   └── index.html         # Frontend HTML
├── requirements.txt       # Python dependencies
├── app.yaml              # App Engine configuration
├── Dockerfile            # Container configuration
├── .gcloudignore        # Files to exclude from deployment
└── DEPLOYMENT_GUIDE.md  # This guide
```

## 🔧 Configuration

### Environment Variables

Add to `app.yaml` if needed:
```yaml
env_variables:
  FLASK_ENV: production
  MODEL_PATH: sentiment_model.pkl
  LOG_LEVEL: INFO
```

### Scaling Configuration

**App Engine** (in `app.yaml`):
```yaml
automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.6
```

**Cloud Run**:
```bash
gcloud run services update sentiment-app \
    --min-instances 0 \
    --max-instances 10 \
    --concurrency 80
```

## 🔍 Monitoring & Logging

### View Logs

**App Engine:**
```bash
gcloud app logs tail -s default
```

**Cloud Run:**
```bash
gcloud run logs tail --service sentiment-app --region us-central1
```

### Performance Monitoring

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to **Monitoring** > **Dashboards**
3. Create custom dashboards for your application

## 🛡️ Security

### Enable HTTPS

Both App Engine and Cloud Run automatically provide HTTPS endpoints.

### Custom Domain (Optional)

1. **Verify domain ownership** in Google Search Console
2. **Map domain** to your app:

```bash
gcloud app domain-mappings create YOUR_DOMAIN.com
```

### API Security

Add authentication if needed:
```python
# In main.py
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.environ.get('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict_sentiment():
    # ... existing code
```

## 💰 Cost Optimization

### App Engine Pricing
- **Free tier**: 28 frontend instance hours per day
- **After free tier**: ~$0.05 per hour per instance

### Cloud Run Pricing
- **Free tier**: 2 million requests per month
- **After free tier**: $0.40 per million requests

### Cost Saving Tips
1. Set appropriate **max instances**
2. Use **min instances = 0** for Cloud Run
3. Implement **request caching** for repeated queries
4. Monitor usage in **Cloud Console**

## 🔄 CI/CD (Optional)

### GitHub Actions Deployment

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Google Cloud

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - uses: google-github-actions/setup-gcloud@v0
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ secrets.GCP_PROJECT_ID }}
    
    - name: Deploy to App Engine
      run: gcloud app deploy app.yaml --quiet
```

## 🧪 Testing

### Health Check

Your app includes a health endpoint:
```
GET https://YOUR_APP_URL/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2023-10-25T16:30:00Z"
}
```

### API Testing

```bash
# Test prediction endpoint
curl -X POST https://YOUR_APP_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

## 🆘 Troubleshooting

### Common Issues

**1. Model file not found:**
```
Error: [Errno 2] No such file or directory: 'sentiment_model.pkl'
```
**Solution:** Ensure `sentiment_model.pkl` is in the project root and not in `.gcloudignore`

**2. Memory errors:**
```
Error: Memory limit exceeded
```
**Solution:** Increase memory in `app.yaml`:
```yaml
resources:
  memory_gb: 4
```

**3. Cold start issues:**
```
Error: Timeout waiting for application to start
```
**Solution:** Increase timeout in `app.yaml`:
```yaml
entrypoint: gunicorn --bind :$PORT --workers 2 --timeout 300 main:app
```

### Debug Commands

```bash
# Check deployment status
gcloud app versions list

# View application details
gcloud app describe

# Stream logs
gcloud app logs tail -s default

# Check service status
gcloud app services list
```

## 📞 Support

- **Google Cloud Support**: [cloud.google.com/support](https://cloud.google.com/support)
- **Documentation**: [cloud.google.com/docs](https://cloud.google.com/docs)
- **Community**: [stackoverflow.com](https://stackoverflow.com/questions/tagged/google-cloud-platform)

## 🎉 Success!

After successful deployment:

1. ✅ Your app is live at: `https://YOUR_PROJECT_ID.uc.r.appspot.com`
2. ✅ Frontend accessible via browser
3. ✅ API endpoints available for integration
4. ✅ Automatic scaling enabled
5. ✅ Monitoring and logging configured

**Next Steps:**
- Share your app URL
- Monitor performance metrics
- Consider adding more features
- Implement user authentication if needed

---

**🚀 Your Sentiment Analysis App is now live on Google Cloud Platform!**