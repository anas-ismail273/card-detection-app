# Railway Deployment Guide

## Quick Railway Setup

### 1. Push to GitHub (if not already done)
```bash
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

### 2. Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository
6. Railway will automatically detect it's a Python app

### 3. Set Environment Variables
In your Railway project dashboard:
- Go to "Variables" tab
- Add: `OPENAI_API_KEY` = your OpenAI API key
- Add: `FLASK_SECRET_KEY` = generate a secure random string
- Add: `ENV` = `production`

### 4. Deploy
- Railway will automatically deploy when you push to GitHub
- First deployment takes 5-10 minutes due to ML dependencies
- Subsequent deployments are faster

## Environment Variables Needed
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
FLASK_SECRET_KEY=your-secure-random-secret-key
ENV=production
```

## Features
✅ **Automatic Environment Detection** - App detects Railway environment
✅ **Dynamic Port Assignment** - Uses Railway's PORT environment variable
✅ **Production Mode** - Disables debug mode in production
✅ **Error Handling** - Graceful fallbacks for missing dependencies
✅ **GitHub Integration** - Auto-deploys on push to main branch

## Monitoring
- Railway dashboard shows logs, metrics, and deployments
- App sleeps after 10-15 minutes of inactivity (saves compute hours)
- Wakes up automatically when accessed (10-30 second delay)

## Troubleshooting
- Check Railway logs for deployment errors
- Ensure all environment variables are set
- Large model files (yolo.pt) might cause longer build times
- OpenAI API costs apply per request

Your app is now ready for Railway deployment!