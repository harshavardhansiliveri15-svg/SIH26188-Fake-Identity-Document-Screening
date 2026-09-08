"""
TAMPERING DETECTION MODULE
For SIH26188 - AI-Powered Fake Identity & Document Screening

This module detects if a document has been tampered with using multiple techniques.
"""

import cv2
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


def detect_tampering(document_image):
    """
    Main function to detect tampering in a document.
    
    Input:
    - document_image: PIL Image object or image path
    
    Output:
    - Dictionary with tampering analysis results
    """
    
    # Convert PIL Image to OpenCV format if needed
    if isinstance(document_image, Image.Image):
        # Convert PIL Image to numpy array
        img_rgb = np.array(document_image)
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        # If it's a file path, read it
        img = cv2.imread(str(document_image))
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Dictionary to store scores from different detection methods
    scores = {}
    
    
    # ============================================================
    # DETECTION METHOD 1: COMPRESSION ARTIFACTS
    # ============================================================
    # Real documents usually have less JPEG compression artifacts
    # Tampered documents often have heavy compression marks
    
    def check_compression():
        """Check for JPEG compression artifacts"""
        
        try:
            # Analyze 8x8 blocks (JPEG standard)
            h, w = gray.shape
            artifact_blocks = 0
            total_blocks = 0
            
            for i in range(0, h-8, 8):
                for j in range(0, w-8, 8):
                    block = gray[i:i+8, j:j+8]
                    
                    # Calculate variance at block edges
                    top_edge = np.var(block[0, :])
                    bottom_edge = np.var(block[-1, :])
                    left_edge = np.var(block[:, 0])
                    right_edge = np.var(block[:, -1])
                    
                    edge_variance = (top_edge + bottom_edge + left_edge + right_edge) / 4
                    
                    # Low variance = compression artifact
                    if edge_variance < 2:
                        artifact_blocks += 1
                    
                    total_blocks += 1
            
            # Calculate percentage
            if total_blocks > 0:
                compression_score = (artifact_blocks / total_blocks) * 100
            else:
                compression_score = 0
            
            return min(compression_score, 100)
        
        except:
            return 0
    
    scores['compression'] = check_compression()
    
    
    # ============================================================
    # DETECTION METHOD 2: BLUR INCONSISTENCY
    # ============================================================
    # Real documents have consistent focus
    # Tampered documents have different blur levels in different areas
    
    def check_blur():
        """Check if some areas are blurry while others are sharp"""
        
        try:
            # Use Laplacian to measure focus/sharpness
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Divide image into 4 regions
            h, w = gray.shape
            top_left = laplacian[0:h//2, 0:w//2]
            top_right = laplacian[0:h//2, w//2:w]
            bottom_left = laplacian[h//2:h, 0:w//2]
            bottom_right = laplacian[h//2:h, w//2:w]
            
            regions = [top_left, top_right, bottom_left, bottom_right]
            
            # Calculate sharpness of each region
            sharpness_values = [np.var(r) for r in regions]
            
            # If regions have very different sharpness = suspicious
            avg_sharpness = np.mean(sharpness_values)
            std_sharpness = np.std(sharpness_values)
            
            # High std = inconsistent focus
            if avg_sharpness > 0:
                blur_score = (std_sharpness / avg_sharpness) * 50
            else:
                blur_score = 0
            
            return min(blur_score, 100)
        
        except:
            return 0
    
    scores['blur'] = check_blur()
    
    
    # ============================================================
    # DETECTION METHOD 3: COLOR CHANNEL ANOMALIES
    # ============================================================
    # Real documents have natural color balance
    # Tampered documents may have unnatural colors
    
    def check_colors():
        """Check if colors look unnatural"""
        
        try:
            # Split into color channels
            b, g, r = cv2.split(img)
            
            # Real images have high correlation between channels
            # Tampered images may have broken correlation
            
            # Flatten to 1D for correlation
            r_flat = r.flatten().astype(np.float32)
            g_flat = g.flatten().astype(np.float32)
            b_flat = b.flatten().astype(np.float32)
            
            # Calculate correlation between channels
            corr_rg = np.corrcoef(r_flat, g_flat)[0, 1]
            corr_rb = np.corrcoef(r_flat, b_flat)[0, 1]
            corr_gb = np.corrcoef(g_flat, b_flat)[0, 1]
            
            # Replace NaN with 0
            corr_rg = 0 if np.isnan(corr_rg) else corr_rg
            corr_rb = 0 if np.isnan(corr_rb) else corr_rb
            corr_gb = 0 if np.isnan(corr_gb) else corr_gb
            
            avg_correlation = (corr_rg + corr_rb + corr_gb) / 3
            
            # Natural images have correlation > 0.7
            # If lower = suspicious
            if avg_correlation < 0.7:
                color_score = (1 - avg_correlation) * 70
            else:
                color_score = 0
            
            return min(color_score, 100)
        
        except:
            return 0
    
    scores['color'] = check_colors()
    
    
    # ============================================================
    # DETECTION METHOD 4: EDGE DISCONTINUITIES
    # ============================================================
    # Copy-paste attacks leave unnatural edges
    
    def check_edges():
        """Check for unnatural edges (copy-paste detection)"""
        
        try:
            # Find edges in image
            edges = cv2.Canny(gray, 50, 150)
            
            # Count edges per row and column
            edge_per_row = np.sum(edges, axis=1)
            edge_per_col = np.sum(edges, axis=0)
            
            # Look for sudden changes
            row_diff = np.abs(np.diff(edge_per_row))
            col_diff = np.abs(np.diff(edge_per_col))
            
            # Find anomalies
            row_std = np.std(row_diff)
            col_std = np.std(col_diff)
            
            row_anomalies = np.sum(row_diff > (np.mean(row_diff) + 2*row_std))
            col_anomalies = np.sum(col_diff > (np.mean(col_diff) + 2*col_std))
            
            total_anomalies = row_anomalies + col_anomalies
            total_possible = len(row_diff) + len(col_diff)
            
            if total_possible > 0:
                edge_score = (total_anomalies / total_possible) * 60
            else:
                edge_score = 0
            
            return min(edge_score, 100)
        
        except:
            return 0
    
    scores['edges'] = check_edges()
    
    
    # ============================================================
    # DETECTION METHOD 5: NOISE PATTERNS
    # ============================================================
    # Tampered areas often have different noise patterns
    
    def check_noise():
        """Check for unnatural noise"""
        
        try:
            # Apply Laplacian to detect high-frequency noise
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Calculate noise level
            noise_level = np.std(laplacian)
            
            # Divide into regions and compare noise
            h, w = gray.shape
            
            regions = [
                gray[0:h//3, 0:w//3],
                gray[0:h//3, w//3:2*w//3],
                gray[0:h//3, 2*w//3:w],
                gray[h//3:2*h//3, 0:w//3],
                gray[h//3:2*h//3, w//3:2*w//3],
                gray[h//3:2*h//3, 2*w//3:w],
                gray[2*h//3:h, 0:w//3],
                gray[2*h//3:h, w//3:2*w//3],
                gray[2*h//3:h, 2*w//3:w],
            ]
            
            noise_levels = [np.std(cv2.Laplacian(r, cv2.CV_64F)) for r in regions]
            
            # High variation in noise = suspicious
            noise_std = np.std(noise_levels)
            noise_mean = np.mean(noise_levels)
            
            if noise_mean > 0:
                noise_score = (noise_std / noise_mean) * 40
            else:
                noise_score = 0
            
            return min(noise_score, 100)
        
        except:
            return 0
    
    scores['noise'] = check_noise()
    
    
    # ============================================================
    # CALCULATE FINAL SCORE
    # ============================================================
    
    # Weighted average - compression and blur are most important
    weights = {
        'compression': 0.30,  # Most reliable
        'blur': 0.25,         # Very good indicator
        'color': 0.15,        # Good indicator
        'edges': 0.20,        # Copy-paste detection
        'noise': 0.10         # Supporting evidence
    }
    
    # Calculate weighted average
    final_score = 0
    for method, score in scores.items():
        final_score += score * weights[method]
    
    final_score = round(final_score, 2)
    
    
    # ============================================================
    # DETERMINE STATUS
    # ============================================================
    
    if final_score < 30:
        status = "LOW"
        message = "✅ No significant tampering detected."
    
    elif final_score < 60:
        status = "MEDIUM"
        message = "⚠️ Some suspicious artifacts detected. Document requires manual review."
    
    else:
        status = "HIGH"
        message = "🚨 Strong evidence of document tampering detected."
    
    
    # ============================================================
    # RETURN RESULTS
    # ============================================================
    
    return {
        "status": status,
        "score": final_score,
        "message": message,
        "detailed_scores": {
            "compression_artifacts": round(scores['compression'], 2),
            "blur_inconsistency": round(scores['blur'], 2),
            "color_anomalies": round(scores['color'], 2),
            "edge_discontinuities": round(scores['edges'], 2),
            "noise_patterns": round(scores['noise'], 2)
        },
        "weights": weights
    }


# ============================================================
# FOR TESTING (run this file directly to test)
# ============================================================

if __name__ == "__main__":
    
    # Test with an image
    test_image_path = "test_image.png"
    
    try:
        result = detect_tampering(test_image_path)
        
        print("=" * 50)
        print("TAMPERING DETECTION RESULTS")
        print("=" * 50)
        print(f"Status: {result['status']}")
        print(f"Score: {result['score']}/100")
        print(f"Message: {result['message']}")
        print(f"\nDetailed Analysis:")
        for method, score in result['detailed_scores'].items():
            print(f"  - {method}: {score}")
        print("=" * 50)
    
    except Exception as e:
        print(f"Error: {e}")
