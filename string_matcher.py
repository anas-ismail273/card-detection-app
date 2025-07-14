# -*- coding: utf-8 -*-
"""String matching module for database search"""

import os
import pandas as pd

# Try to import jiwer with proper error handling
try:
    import jiwer
except ImportError:
    print("Warning: jiwer not installed. Please install it with: pip install jiwer")
    jiwer = None

HOME = os.getcwd()
print(f"Working directory: {HOME}")

class OCRStringMatcher:
    def __init__(self, ground_truth_df, transform=None):
        """
        Initialize with the ground truth dataframe and optional jiwer transform.
        """
        self.ground_truth_df = ground_truth_df.reset_index(drop=True)
        
        if jiwer is None:
            # Create a simple fallback transform
            self.transform = self._simple_transform
        else:
            self.transform = transform or jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveWhiteSpace(replace_by_space=True),
                jiwer.Strip()
            ])
        self.results = []

    def _simple_transform(self, text):
        """Simple text transformation when jiwer is not available"""
        if not text:
            return ""
        import re
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip whitespace
        text = text.strip()
        return text

    def _simple_wer(self, reference, hypothesis):
        """Simple WER calculation when jiwer is not available"""
        if not reference and not hypothesis:
            return 0.0
        if not reference:
            return 1.0
        if not hypothesis:
            return 1.0
            
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Simple edit distance approximation
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0
            
        # Calculate simple word overlap
        ref_set = set(ref_words)
        hyp_set = set(hyp_words)
        
        intersection = len(ref_set & hyp_set)
        union = len(ref_set | hyp_set)
        
        if union == 0:
            return 0.0
            
        # Simple approximation: 1 - (intersection / max(len(ref), len(hyp)))
        max_len = max(len(ref_words), len(hyp_words))
        return 1.0 - (intersection / max_len)

    def _simple_cer(self, reference, hypothesis):
        """Simple CER calculation when jiwer is not available"""
        if not reference and not hypothesis:
            return 0.0
        if not reference:
            return 1.0
        if not hypothesis:
            return 1.0
            
        # Simple character-level comparison
        ref_chars = set(reference.lower())
        hyp_chars = set(hypothesis.lower())
        
        intersection = len(ref_chars & hyp_chars)
        max_len = max(len(reference), len(hypothesis))
        
        if max_len == 0:
            return 0.0
            
        return 1.0 - (intersection / max_len)

    @staticmethod
    def combine_ocr_text(ocr_result):
        """
        Combine OCR text fragments for each image.
        Returns dict of {image_name: full_text}
        """
        combined = {}
        for image_name, tokens in ocr_result.items():
            if isinstance(tokens, list):
                full_text = " ".join([t.get('text', '') for t in tokens if isinstance(t, dict)])
            else:
                full_text = str(tokens)
            combined[image_name] = full_text
        return combined

    def evaluate(self, ocr_texts):
        """
        Evaluate OCR texts against ground truth data.
        ocr_texts: dict of {image_name: ocr_text}
        """
        self.results.clear()

        for image_name, ocr_text in ocr_texts.items():
            print(f"\nProcessing {image_name}...")
            print(f"OCR Text: {ocr_text}")

            best_match = None
            best_index = None
            best_wer = None
            best_cer = None
            best_gt_combined = None

            ocr_transformed = self.transform(ocr_text)
            print(f"Transformed OCR: {ocr_transformed}")

            for idx, row in self.ground_truth_df.iterrows():
                # Concatenate all relevant fields in the ground truth row
                gt_combined = " ".join(
                    str(val).strip()
                    for val in row.values
                    if pd.notnull(val) and str(val).strip().lower() not in ("n/a", "")
                )

                if not gt_combined.strip():
                    continue

                gt_transformed = self.transform(gt_combined)

                if jiwer is not None:
                    try:
                        wer = jiwer.wer(gt_transformed, ocr_transformed)
                        cer = jiwer.cer(gt_transformed, ocr_transformed)
                    except:
                        # Fallback to simple calculation
                        wer = self._simple_wer(gt_transformed, ocr_transformed)
                        cer = self._simple_cer(gt_transformed, ocr_transformed)
                else:
                    wer = self._simple_wer(gt_transformed, ocr_transformed)
                    cer = self._simple_cer(gt_transformed, ocr_transformed)

                wer = min(wer, 1.0)
                cer = min(cer, 1.0)

                if best_wer is None or wer < best_wer:
                    best_match = row
                    best_index = idx
                    best_wer = wer
                    best_cer = cer
                    best_gt_combined = gt_combined

            # Only accept matches with high confidence
            # WER ≤ 0.2 means at least 80% similarity
            if best_match is None or best_wer > 0.2:
                print(f"  ❌ No good matches found for '{ocr_text}' (best WER: {best_wer:.2f} - need ≤ 0.20 for 80% match)")
                self.results.append({
                    "image": image_name,
                    "ground_truth_idx": None,
                    "WER": 1.0,
                    "CER": 1.0,
                    "gt_text": "Card not found in database",
                    "ocr_text": ocr_text,
                    "match_quality": "not_found"
                })
                continue

            print(f"  ✅ Good match found: {best_gt_combined} (WER: {best_wer:.2f} - {((1-best_wer)*100):.1f}% similarity)")
            
            self.results.append({
                "image": image_name,
                "ground_truth_idx": best_index,
                "WER": best_wer,
                "CER": best_cer,
                "gt_text": best_gt_combined,
                "ocr_text": ocr_text,
                "match_quality": "good" if best_wer <= 0.1 else "acceptable"
            })

        return pd.DataFrame(self.results)

    def print_report(self, results_df=None):
        """
        Print detailed OCR evaluation report.
        """
        if results_df is None:
            if not self.results:
                print("No results to display.")
                return
            results_df = pd.DataFrame(self.results)

        print("\n🗂️ === OCR Evaluation Report ===\n")

        for _, match in results_df.iterrows():
            idx = match["ground_truth_idx"]
            gt_row = self.ground_truth_df.loc[idx]

            print("=" * 60)
            print(f"IMAGE: {match['image']}")
            print(f"WER: {match['WER']:.2%}")
            print(f"CER: {match['CER']:.2%}\n")

            print(f"GAME: {gt_row.get('game_name', 'N/A')}")
            print(f"CARD TITLE: {gt_row.get('card_title', 'N/A')}")
            print(f"TAGS: {gt_row.get('tags', 'N/A')}")
            print(f"FIGURE: {gt_row.get('figure_name', 'N/A')}")
            print(f"POWER: {gt_row.get('power', 'N/A')}")
            print(f"SUIT: {gt_row.get('suit', 'N/A')}")
            print(f"RANK: {gt_row.get('rank', 'N/A')}")
            print(f"EXTRA NOTES: {gt_row.get('extra_notes', 'N/A')}")

            desc = gt_row.get('description', '')
            if pd.notnull(desc) and desc.strip():
                print("\nDESCRIPTION:")
                print(f"  {desc.strip()}")

            print("=" * 60)
            print("\n")