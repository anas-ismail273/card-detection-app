from flask import Flask, render_template, request, jsonify, session, send_from_directory
import os
import json
import tempfile
from werkzeug.utils import secure_filename
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
import base64
import io

# Environment detection and secret loading
def load_environment_secrets():
    """Load secrets based on environment (development vs production)"""
    
    # Check if we're in a production environment (GitHub Actions, Heroku, etc.)
    is_production = (
        os.getenv('GITHUB_ACTIONS') == 'true' or  # GitHub Actions
        os.getenv('HEROKU') is not None or        # Heroku
        os.getenv('RAILWAY_ENVIRONMENT') is not None or  # Railway
        os.getenv('VERCEL') is not None or        # Vercel
        os.getenv('FLASK_ENV') == 'production' or # Manual production flag
        os.getenv('ENV') == 'production'          # Generic production flag
    )
    
    if is_production:
        print("🚀 Production environment detected - loading secrets from environment variables")
        # In production, secrets should be available as environment variables
        # (set via GitHub repository secrets, Heroku config vars, etc.)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ Warning: OPENAI_API_KEY not found in production environment")
        else:
            print("✅ OpenAI API key loaded from environment variables")
        return api_key
    else:
        print("🛠️  Development environment detected - loading secrets from .env file")
        # In development, try to load from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("❌ Warning: OPENAI_API_KEY not found in .env file")
            else:
                print("✅ OpenAI API key loaded from .env file")
            return api_key
        except ImportError:
            print("❌ Warning: python-dotenv not installed. Install with: pip install python-dotenv")
            return os.getenv('OPENAI_API_KEY')

# Load API key based on environment
OPENAI_API_KEY = load_environment_secrets()

# Set the API key in environment for the OpenAI library
if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Import the existing modules
from ocr import YoloMasker, OCRProcessor, LLMProcessor
from string_matcher import OCRStringMatcher

app = Flask(__name__)

# Use a proper secret key (different from API key)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-for-sessions-change-this-in-production')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MODEL_PATH = 'yolo.pt'
DATABASE_PATH = 'data_latest.csv'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the database with error handling for CSV parsing issues
try:
    database_df = pd.read_csv(DATABASE_PATH, on_bad_lines='skip', quotechar='"', escapechar='\\')
    print(f"Successfully loaded {len(database_df)} cards from database")
except Exception as e:
    print(f"Error loading database: {e}")
    # Create a minimal fallback database
    database_df = pd.DataFrame({
        'game_name': ['Demo'],
        'card_title': ['Sample Card'],
        'description': ['Sample description'],
        'tags': ['Sample'],
        'figure_name': ['Sample'],
        'power': ['1/1'],
        'suit': ['Hearts'],
        'rank': ['A'],
        'extra_notes': ['Demo card']
    })

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Store the filepath in session for later use
        session['uploaded_file'] = filepath
        
        return jsonify({
            'success': True, 
            'filename': filename,
            'image_url': f'/uploaded_image/{filename}'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/uploaded_image/<filename>')
def uploaded_image(filename):
    """Serve uploaded images"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/process', methods=['POST'])
def process_image():
    if 'uploaded_file' not in session:
        return jsonify({'error': 'No file uploaded'}), 400
    
    image_path = session['uploaded_file']
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Uploaded file not found'}), 404
    
    try:
        # Check if YOLO model exists
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': f'YOLO model not found at {MODEL_PATH}. Please ensure the model file exists.'}), 500
        
        print(f"Processing image: {image_path}")
        
        # Initialize the YOLO masker
        masker = YoloMasker(MODEL_PATH)
        
        # Process the image to get masked individual cards
        masked_image_paths = masker.process(image_path, conf=0.75)
        
        if not masked_image_paths:
            return jsonify({'error': 'No cards detected in the image. Try uploading an image with clearly visible cards.'}), 400
        
        print(f"Detected {len(masked_image_paths)} cards")
        
        # Initialize OCR processor
        ocr_processor = OCRProcessor(lang='en')
        
        # Perform OCR on the masked images
        ocr_results = ocr_processor.perform_ocr_on_images(masked_image_paths)
        
        # Initialize LLM processor (now with API key)
        llm_processor = LLMProcessor(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Perform LLM text extraction (with better error handling)
        try:
            llm_results = llm_processor.perform_text_extraction_on_images(masked_image_paths)
            print("LLM processing completed successfully")
        except Exception as llm_error:
            print(f"LLM processing failed: {llm_error}")
            # Create fallback LLM results with same order as OCR
            llm_results = {}
            for img_path in masked_image_paths:
                llm_results[img_path] = [{
                    "text": f"LLM processing failed: {str(llm_error)}",
                    "confidence": 0.0,
                    "bbox": [],
                    "model_used": "N/A",
                    "error": str(llm_error)
                }]
        
        # Format the results for display - ensuring same order for both OCR and LLM
        ocr_formatted = {}
        llm_formatted = {}
        
        # Process results in the same order as masked_image_paths to ensure consistency
        for idx, img_path in enumerate(masked_image_paths):
            # Extract text from OCR results
            ocr_text = ""
            if img_path in ocr_results and ocr_results[img_path]:
                ocr_text = " ".join([item.get('text', '') for item in ocr_results[img_path] if item.get('text')])
            
            # Extract text from LLM results  
            llm_text = ""
            if img_path in llm_results and llm_results[img_path]:
                llm_text = llm_results[img_path][0].get('text', '')
            
            # Use the base filename without extension as a cleaner identifier
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Remove the original filename prefix to get just the mask identifier
            if '_mask_' in base_name:
                card_identifier = f"Card {base_name.split('_mask_')[-1]}"
            else:
                card_identifier = f"Card {idx + 1}"
            
            ocr_formatted[card_identifier] = ocr_text if ocr_text else 'No text detected'
            llm_formatted[card_identifier] = llm_text if llm_text else 'No text detected'
        
        # Store results in session for potential database search
        session['ocr_results'] = ocr_results
        session['masked_images'] = masked_image_paths
        
        return jsonify({
            'success': True,
            'ocr_results': ocr_formatted,
            'llm_results': llm_formatted,
            'num_cards': len(masked_image_paths),
            'message': f'Successfully processed {len(masked_image_paths)} cards'
        })
        
    except Exception as e:
        print(f"Processing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}. Please check if all required dependencies are installed.'}), 500

@app.route('/search_database', methods=['POST'])
def search_database():
    if 'ocr_results' not in session:
        return jsonify({'error': 'No OCR results available. Please run processing first.'}), 400
    
    try:
        # Initialize the string matcher with the database
        matcher = OCRStringMatcher(database_df)
        
        # Fix: Convert OCR results to the format expected by combine_ocr_text
        # The combine_ocr_text method expects: {image_name: [{'text': 'text1'}, {'text': 'text2'}]}
        # But we have: {image_path: [{'text': 'text1', 'confidence': 0.9, 'bbox': []}]}
        
        formatted_ocr_results = {}
        for image_path, ocr_data in session['ocr_results'].items():
            # Use just the filename as the key for cleaner display
            image_name = os.path.basename(image_path)
            formatted_ocr_results[image_name] = ocr_data
        
        # Combine OCR text for matching
        ocr_combined = matcher.combine_ocr_text(formatted_ocr_results)
        
        # Debug: Print the combined text to see what we're matching
        print("OCR combined text for matching:")
        for key, text in ocr_combined.items():
            print(f"  {key}: {text}")
        
        # Perform the database search
        results_df = matcher.evaluate(ocr_combined)
        
        if results_df.empty:
            return jsonify({
                'success': False,
                'error': 'No matches found in the database. The extracted text may not match any cards in our database.'
            }), 400
        
        # Format results for display
        search_results = {}
        
        for idx, (_, row) in enumerate(results_df.iterrows()):
            # Use the actual image name from the results
            card_name = row['image']
            
            # Get the corresponding database entry
            db_entry = database_df.iloc[row['ground_truth_idx']]
            
            # Format the database match information
            match_info = {
                'card_title': str(db_entry.get('card_title', 'N/A')),
                'game_name': str(db_entry.get('game_name', 'N/A')),
                'description': str(db_entry.get('description', 'N/A')),
                'tags': str(db_entry.get('tags', 'N/A')),
                'figure_name': str(db_entry.get('figure_name', 'N/A')),
                'power': str(db_entry.get('power', 'N/A')),
                'suit': str(db_entry.get('suit', 'N/A')),
                'rank': str(db_entry.get('rank', 'N/A')),
                'extra_notes': str(db_entry.get('extra_notes', 'N/A')),
                'wer': f"{row['WER']:.2%}",
                'cer': f"{row['CER']:.2%}",
                'match_confidence': f"{(1 - row['WER']) * 100:.1f}%"
            }
            
            search_results[card_name] = match_info
        
        return jsonify({
            'success': True,
            'search_results': search_results,
            'num_matches': len(results_df)
        })
        
    except Exception as e:
        print(f"Database search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Database search failed: {str(e)}'}), 500

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear session data when page is refreshed"""
    session.clear()
    
    # Clean up uploaded files
    if os.path.exists(UPLOAD_FOLDER):
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    
    # Clean up masked outputs
    if os.path.exists('masked_outputs'):
        for file in os.listdir('masked_outputs'):
            file_path = os.path.join('masked_outputs', file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)