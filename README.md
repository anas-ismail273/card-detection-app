# Card Detection AI Lab

A Flask-based web application for detecting and extracting text from trading cards using YOLO object detection, OCR, and LLM technologies.

## Features

- **Card Detection**: Uses YOLO model to detect and segment individual cards from images
- **Text Extraction**: Dual approach using PaddleOCR and OpenAI's GPT-4 Vision API
- **Database Matching**: Matches extracted text against a card database using string similarity
- **Web Interface**: User-friendly Flask web application

## Setup Instructions

### 1. Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anas-ismail273/card-detection-app.git
   cd card-detection-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

### 2. GitHub Repository Secrets Setup

For production deployment and GitHub Actions, set up repository secrets:

1. **Go to your GitHub repository**: Navigate to `https://github.com/anas-ismail273/card-detection-app`

2. **Access repository settings**: Click on "Settings" → "Secrets and variables" → "Actions"

3. **Add repository secret**:
   - **Name**: `OPENAI_API_KEY`
   - **Secret**: Your actual OpenAI API key

4. **Push your code**: The GitHub Actions workflow will automatically use the repository secret

## Environment Variable Priority

The application loads API keys in the following order:

1. **GitHub Actions Environment**: When running in GitHub Actions, it uses `${{ secrets.OPENAI_API_KEY }}`
2. **Local .env file**: For local development, it loads from `.env` file
3. **System environment**: Falls back to system environment variables
4. **Graceful degradation**: If no API key is found, LLM processing is skipped

## File Structure

```
├── app.py                 # Main Flask application
├── ocr.py                 # OCR and LLM processing modules
├── string_matcher.py      # Database matching functionality
├── requirements.txt       # Python dependencies
├── yolo.pt               # YOLO model file
├── data_latest.csv       # Card database
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore file
├── templates/
│   └── index.html        # Web interface template
├── uploads/              # Uploaded images directory
├── masked_outputs/       # Processed card images
└── .github/
    └── workflows/
        └── deploy.yml    # GitHub Actions workflow
```

## Usage

1. **Upload an image**: Use the web interface to upload an image containing trading cards
2. **Process the image**: The system will:
   - Detect individual cards using YOLO
   - Extract text using both OCR and LLM
   - Display results for comparison
3. **Search database**: Match extracted text against the card database
4. **View results**: See matching cards with confidence scores

## Technologies Used

- **Backend**: Flask (Python)
- **Computer Vision**: YOLO (Ultralytics), OpenCV
- **OCR**: PaddleOCR
- **LLM**: OpenAI GPT-4 Vision API
- **Database**: CSV with pandas
- **String Matching**: Jiwer (WER/CER metrics)
- **Frontend**: HTML, CSS, JavaScript

## Security Features

- ✅ API keys stored securely in environment variables
- ✅ GitHub repository secrets for production
- ✅ .env file excluded from version control
- ✅ Secure file uploads with validation
- ✅ Session management for temporary data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.