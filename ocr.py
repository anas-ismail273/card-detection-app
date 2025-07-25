# -*- coding: utf-8 -*-
"""OCR module for card detection and text extraction"""

import torch
from PIL import Image
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Environment detection and secret loading
def load_environment_secrets():
    """Load secrets based on environment (development vs production)"""
    
    # Check if we're in a production environment
    is_production = (
        os.getenv('GITHUB_ACTIONS') == 'true' or  # GitHub Actions
        os.getenv('HEROKU') is not None or        # Heroku
        os.getenv('RAILWAY_ENVIRONMENT') is not None or  # Railway
        os.getenv('VERCEL') is not None or        # Vercel
        os.getenv('FLASK_ENV') == 'production' or # Manual production flag
        os.getenv('ENV') == 'production'          # Generic production flag
    )
    
    if is_production:
        # In production, secrets should be available as environment variables
        return os.getenv('OPENAI_API_KEY')
    else:
        # In development, try to load from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            return os.getenv('OPENAI_API_KEY')
        except ImportError:
            return os.getenv('OPENAI_API_KEY')

# Load API key based on environment
OPENAI_API_KEY = load_environment_secrets()

# Try to import the required libraries with proper error handling
YOLO_AVAILABLE = True
try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Please install it with: pip install ultralytics")
    YOLO_AVAILABLE = False
    
try:
    from paddleocr import PaddleOCR
except ImportError:
    print("Warning: paddleocr not installed. Please install it with: pip install paddleocr paddlepaddle")

try:
    import openai
    import base64
except ImportError:
    print("Warning: openai not installed. Please install it with: pip install openai")

HOME = os.getcwd()
print(f"Working directory: {HOME}")

class YoloMasker:
    def __init__(self, model_path, output_dir="masked_outputs"):
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO is not available. Please ensure ultralytics is installed.")
        
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {str(e)}")
            
        self.output_dir = output_dir
        self.result = None
        os.makedirs(self.output_dir, exist_ok=True)

    def predict(self, image_path, conf=0.75):
        image = Image.open(image_path).convert("RGB")
        results = self.model.predict(image, conf=conf)
        self.result = results[0]
        return results[0]

    def apply_masking(self, image_path, yolo_result):
        from PIL import ImageDraw
        
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        base_filename = Path(image_path).stem
        os.makedirs(self.output_dir, exist_ok=True)

        if not hasattr(yolo_result, "masks") or yolo_result.masks is None:
            raise ValueError("No segmentation masks found. Use a segmentation-capable YOLO model.")

        masked_images = []
        for idx, segment in enumerate(yolo_result.masks.xy):
            # 1) Make a full‐size mask
            mask = Image.new("L", image_pil.size, 0)
            poly = [(x, y) for x, y in segment]
            ImageDraw.Draw(mask).polygon(poly, outline=1, fill=255)
            mask_np = np.array(mask)

            # 2) Compute bounding box of the mask
            ys, xs = np.where(mask_np == 255)
            if len(xs) == 0 or len(ys) == 0:
                continue
                
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            # 3) Crop both the original and mask to that box
            cropped_image = image_pil.crop((x_min, y_min, x_max, y_max))
            cropped_mask = mask.crop((x_min, y_min, x_max, y_max))
            cm_np = np.array(cropped_mask)
            ci_np = np.array(cropped_image)

            # 4) Apply mask to the crop
            zoomed_np = np.zeros_like(ci_np)
            zoomed_np[cm_np == 255] = ci_np[cm_np == 255]
            zoomed_pil = Image.fromarray(zoomed_np)

            # 5) Save it
            out_path = os.path.join(self.output_dir, f"{base_filename}_mask_{idx}.png")
            zoomed_pil.save(out_path)
            masked_images.append(out_path)

        return masked_images

    def show_result(self):
        if self.result:
            self.result.show()

    def process(self, image_path, conf=0.75):
        result = self.predict(image_path, conf)
        return self.apply_masking(image_path, result)

class OCRProcessor:
    def __init__(self, lang='en', use_textline_orientation=True):
        self.ocr = PaddleOCR(use_textline_orientation=use_textline_orientation, lang=lang)
        self.results = {}

    def perform_ocr_on_images(self, image_paths):
        self.results = {}

        for img_path in image_paths:
            print(f"Processing OCR for: {img_path}")
            try:
                ocr_result = self.ocr.ocr(img_path)
                
                # Debug: print raw OCR result structure
                print(f"Raw OCR result type: {type(ocr_result)}")
                
                if not ocr_result or not isinstance(ocr_result, list) or not ocr_result[0]:
                    print("No OCR results found")
                    self.results[img_path] = []
                    continue

                extracted = []
                result_data = ocr_result[0]
                
                # Check if this is the new PaddleOCR format with rec_texts, rec_scores, rec_polys
                if isinstance(result_data, dict) and 'rec_texts' in result_data:
                    print(f"Using new PaddleOCR format")
                    
                    rec_texts = result_data.get('rec_texts', [])
                    rec_scores = result_data.get('rec_scores', [])
                    rec_polys = result_data.get('rec_polys', [])
                    
                    print(f"Found {len(rec_texts)} texts in {img_path}")
                    
                    # Process each detected text
                    for i, text in enumerate(rec_texts):
                        if i < len(rec_scores):
                            confidence = rec_scores[i]
                        else:
                            confidence = 0.0
                            
                        if i < len(rec_polys):
                            bbox_array = rec_polys[i]
                        else:
                            bbox_array = None
                        
                        print(f"Text: '{text}', Confidence: {confidence}")
                        
                        # Convert bbox array to the expected format
                        bbox = []
                        if bbox_array is not None:
                            try:
                                # bbox_array is numpy array, convert to list of [x, y] points
                                bbox = bbox_array.tolist()
                                print(f"BBox: {bbox}")
                            except Exception as e:
                                print(f"Error converting bbox: {e}")
                                bbox = []
                        
                        extracted.append({
                            "text": text,
                            "confidence": float(confidence),
                            "bbox": bbox
                        })
                
                else:
                    # Handle old format (list of [bbox, (text, confidence)])
                    print(f"Using old PaddleOCR format")
                    text_lines = result_data
                    print(f"Found {len(text_lines)} text lines in {img_path}")
                    
                    for line in text_lines:
                        if line and len(line) >= 2:
                            print(f"Processing line: {line}")
                            
                            # PaddleOCR returns: [bbox, (text, confidence)]
                            bbox = line[0]
                            text_info = line[1]
                            
                            # Extract text and confidence
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = text_info[0]
                                conf = text_info[1]
                            else:
                                text = str(text_info)
                                conf = 0.0
                            
                            print(f"Text: '{text}', Confidence: {conf}")
                            print(f"BBox type: {type(bbox)}, BBox: {bbox}")
                            
                            # Validate and convert bounding box coordinates
                            try:
                                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                                    # Convert coordinates to float, handling potential string values
                                    converted_bbox = []
                                    for pt in bbox:
                                        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                                            # More robust number validation
                                            try:
                                                x = float(pt[0])
                                                y = float(pt[1])
                                                converted_bbox.append([x, y])
                                            except (ValueError, TypeError):
                                                print(f"Invalid coordinate values: {pt}")
                                                converted_bbox.append([0.0, 0.0])
                                        else:
                                            print(f"Invalid point format: {pt}")
                                            converted_bbox.append([0.0, 0.0])
                                    
                                    if len(converted_bbox) >= 4:
                                        extracted.append({
                                            "text": text,
                                            "confidence": float(conf) if isinstance(conf, (int, float)) else 0.0,
                                            "bbox": converted_bbox
                                        })
                                    else:
                                        print(f"Insufficient bbox points: {converted_bbox}")
                                else:
                                    print(f"Invalid bbox format: {bbox}")
                                    
                            except (ValueError, TypeError) as e:
                                print(f"Error converting bbox coordinates: {e}")
                                print(f"Problematic bbox: {bbox}")
                                # Skip this detection if bbox conversion fails
                                continue

                self.results[img_path] = extracted
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                self.results[img_path] = []

        return self.results

    def get_summary_table(self):
        summary = []

        for img_path, texts in self.results.items():
            all_texts = []
            all_confs = []

            for entry in texts:
                all_texts.append(entry["text"])
                all_confs.append(entry["confidence"])

            combined_text = " ".join(all_texts)
            avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

            summary.append({
                "image": os.path.basename(img_path),
                "full_text": combined_text,
                "average_confidence": round(avg_conf, 4)
            })

        df = pd.DataFrame(summary)
        return df.sort_values(by="average_confidence", ascending=False).reset_index(drop=True)

class LLMProcessor:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        # 1. If the user passes a key directly, use it.
        if api_key:
            self.api_key = api_key
        # 2. Otherwise, look for the standard OPENAI_API_KEY env var
        elif os.getenv("OPENAI_API_KEY"):
            self.api_key = os.getenv("OPENAI_API_KEY")
        # 3. If you still don't have a key, create a dummy processor
        else:
            print("Warning: No OpenAI API key found. LLM processing will be skipped.")
            self.client = None
            self.model = model
            self.results = {}
            return

        # Initialize OpenAI client with the API key
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.results = {}

    def encode_image(self, image_path):
        """Encode image to base64 for OpenAI API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def extract_text_from_image(self, image_path):
        """Extract text from a single image using OpenAI's vision model."""
        if not self.client:
            return {
                "text": "LLM processing unavailable (no API key)",
                "confidence": 0.0,
                "model_used": self.model
            }
            
        try:
            b64 = self.encode_image(image_path)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Extract only the visible text from the attached "
                                    "card image. "
                                    "Return exactly the card name, mana cost, type line, "
                                    "rules text, flavor text, and any other visible text, "
                                    "as plain text with no extra commentary or formatting."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=500
            )

            # Raw from the model
            extracted = response.choices[0].message.content

            # Clean up the text
            cleaned = extracted.replace("\\n", "\n").strip()
            cleaned = cleaned.replace("\n", " ")

            confidence = min(0.95, len(cleaned) / 100) if cleaned else 0.0

            return {
                "text": cleaned,
                "confidence": confidence,
                "model_used": self.model
            }

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return {
                "text": f"Error: {str(e)}",
                "confidence": 0.0,
                "model_used": self.model,
                "error": str(e)
            }

    def perform_text_extraction_on_images(self, image_paths):
        """Process multiple images and extract text using LLM."""
        self.results = {}
        for img_path in image_paths:
            print(f"Processing LLM text extraction for: {img_path}")
            result = self.extract_text_from_image(img_path)
            self.results[img_path] = [{
                "text": result["text"],
                "confidence": result["confidence"],
                "bbox": [],  # no bounding boxes from LLM
                "model_used": result["model_used"],
                "error": result.get("error")
            }]
        return self.results

    def get_summary_table(self):
        """Generate summary table of extractions."""
        summary = []
        for img_path, extractions in self.results.items():
            texts = [e["text"] for e in extractions if e["text"]]
            confs = [e["confidence"] for e in extractions if e["text"]]
            combined = " ".join(texts)
            avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.0
            summary.append({
                "image": os.path.basename(img_path),
                "full_text": combined,
                "average_confidence": avg_conf
            })

        df = pd.DataFrame(summary)
        return df.sort_values(
            by="average_confidence", ascending=False
        ).reset_index(drop=True)