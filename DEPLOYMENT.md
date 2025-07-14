# Card Detection AI Lab - Deployment Guide

## Environment Configuration

This application automatically detects whether it's running in development or production and loads secrets accordingly:

### Development Environment (Local)
- Uses `.env` file for API keys
- Loads secrets with `python-dotenv`
- Shows debug information

### Production Environment (Hosting)
- Uses environment variables directly
- Detects production via environment flags
- Uses GitHub repository secrets when deployed

## Setup Instructions

### 1. Development Setup

1. **Clone and install dependencies**:
   ```bash
   git clone https://github.com/anas-ismail273/card-detection-app.git
   cd card-detection-app
   pip install -r requirements.txt
   ```

2. **Create environment file**:
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` file**:
   ```bash
   OPENAI_API_KEY=sk-proj-your-actual-api-key-here
   FLASK_SECRET_KEY=your-secure-random-secret-key-here
   ```

4. **Run locally**:
   ```bash
   python app.py
   ```

### 2. Production Setup (GitHub Repository Secrets)

1. **Go to your GitHub repository settings**:
   - Navigate to: `https://github.com/anas-ismail273/card-detection-app`
   - Click "Settings" → "Secrets and variables" → "Actions"

2. **Add repository secrets**:
   - Name: `OPENAI_API_KEY`
   - Value: Your actual OpenAI API key

3. **Optional: Add Flask secret key**:
   - Name: `FLASK_SECRET_KEY`
   - Value: Generate a secure random string

### 3. Hosting Platform Configuration

#### Heroku
```bash
heroku config:set OPENAI_API_KEY=your-api-key
heroku config:set FLASK_SECRET_KEY=your-secret-key
heroku config:set ENV=production
```

#### Railway
```bash
railway variables set OPENAI_API_KEY=your-api-key
railway variables set FLASK_SECRET_KEY=your-secret-key
railway variables set ENV=production
```

#### Vercel
Add environment variables in your Vercel dashboard:
- `OPENAI_API_KEY`
- `FLASK_SECRET_KEY`
- `ENV=production`

## Environment Detection Logic

The application automatically detects the environment using these indicators:

**Production Environment Detected When:**
- `GITHUB_ACTIONS=true` (GitHub Actions)
- `HEROKU` environment variable exists
- `RAILWAY_ENVIRONMENT` environment variable exists
- `VERCEL` environment variable exists
- `FLASK_ENV=production`
- `ENV=production`

**Development Environment:**
- None of the above conditions are met
- Loads from `.env` file using `python-dotenv`

## Security Features

✅ **API keys never hardcoded in source code**
✅ **Environment-specific secret loading**
✅ **`.env` file excluded from version control**
✅ **GitHub repository secrets for CI/CD**
✅ **Automatic environment detection**

## Testing the Configuration

### Local Development Test
```bash
python -c "
from app import load_environment_secrets
api_key = load_environment_secrets()
if api_key:
    print('✅ API key loaded successfully')
else:
    print('❌ API key not found')
"
```

### Production Test (via GitHub Actions)
The GitHub Actions workflow automatically tests the environment setup when you push to the main branch.

## File Structure

```
├── app.py                    # Main Flask app with environment detection
├── ocr.py                    # OCR module with environment-aware secrets
├── .env.example              # Environment template
├── .env                      # Your local environment (not in git)
├── .gitignore                # Excludes .env and sensitive files
├── requirements.txt          # Python dependencies
├── .github/workflows/deploy.yml  # GitHub Actions workflow
└── README.md                 # This file
```

## Deployment Flow

1. **Development**: Code locally using `.env` file
2. **Push to GitHub**: Code automatically tested with repository secrets
3. **Deploy to hosting**: Platform uses environment variables
4. **Automatic environment detection**: No code changes needed

Your app is now ready for both development and production environments!