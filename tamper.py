"""
TAMPERING DETECTION MODULE V2
For SIH26188 - AI-Powered Fake Identity & Document Screening

Improved with more aggressive detection thresholds for documents
"""

import cv2
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


def detect_tampering(document_image):
    """
    Enhanced tampering detection with aggressive thresholds
    """
    
    # Convert PIL Image to OpenCV format
    if isinstance(document_image, Image.Image):
        img_rgb = np.array(document_image)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(str(document_image))
        if img is None:
            return {
                "status": "ERROR",
                "score": 0,
                "message": "Could not read image"
            }
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    scores = {}
    
    # ============================================================
    # METHOD 1: COMPRESSION ARTIFACTS (More Aggressive)
    # ============================================================
    
    def check_compression():
        try:
            h, w = gray.shape
            artifact_blocks = 0
            total_blocks = 0
            
            # Check 8x8 blocks for JPEG artifacts
            for i in range(0, h-8, 8):
                for j in range(0, w-8, 8):
                    block = gray[i:i+8, j:j+8]
                    
                    # More aggressive: check for ANY edge variance < 5
                    edges_variance = []
                    edges_variance.append(np.var(block[0, :]))     # top
                    edges_variance.append(np.var(block[-1, :]))    # bottom
                    edges_variance.append(np.var(block[:, 0]))     # left
                    edges_variance.append(np.var(block[:, -1]))    # right
                    
                    avg_edge_var = np.mean(edges_variance)
                    
                    # LOWER THRESHOLD = MORE SENSITIVE
                    if avg_edge_var < 5:  # Changed from 2 to 5
                        artifact_blocks += 1
                    
                    total_blocks += 1
            
            if total_blocks > 0:
                compression_score = (artifact_blocks / total_blocks) * 100
            else:
                compression_score = 0
            
            return min(compression_score, 100)
        except:
            return 0
    
    scores['compression'] = check_compression()
    
    
    # ============================================================
    # METHOD 2: BLUR AND FOCUS INCONSISTENCY (More Aggressive)
    # ============================================================
    
    def check_blur():
        try:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_abs = np.abs(laplacian)
            
            # Divide into 9 regions (3x3 grid for better detection)
            h, w = gray.shape
            h_step = h // 3
            w_step = w // 3
            
            regions = []
            for i in range(3):
                for j in range(3):
                    region = laplacian_abs[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                    regions.append(region)
            
            # Calculate variance (sharpness) for each region
            sharpness_values = [np.var(r) for r in regions]
            
            # MORE AGGRESSIVE: Higher sensitivity to focus differences
            avg_sharpness = np.mean(sharpness_values)
            std_sharpness = np.std(sharpness_values)
            max_sharpness = np.max(sharpness_values)
            min_sharpness = np.min(sharpness_values)
            
            # Coefficient of variation
            if avg_sharpness > 0:
                cv = (std_sharpness / avg_sharpness) * 100
                # Also check max-min difference
                max_min_ratio = max_sharpness / (min_sharpness + 1)
                
                # Combine both metrics
                blur_score = (cv * 0.6) + (min(max_min_ratio * 20, 100) * 0.4)
            else:
                blur_score = 0
            
            return min(blur_score, 100)
        except:
            return 0
    
    scores['blur'] = check_blur()
    
    
    # ============================================================
    # METHOD 3: HISTOGRAM ANALYSIS (New - very good for tampering)
    # ============================================================
    
    def check_histogram():
        try:
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            # Real documents have relatively smooth histograms
            # Tampered documents have peaks and valleys
            
            # Check for unnatural peaks
            diffs = np.diff(hist)
            sharp_changes = np.sum(np.abs(diffs) > 0.01)
            
            # Score based on number of sharp changes
            histogram_score = min(sharp_changes * 2, 100)
            
            return histogram_score
        except:
            return 0
    
    scores['histogram'] = check_histogram()
    
    
    # ============================================================
    # METHOD 4: FREQUENCY DOMAIN ANALYSIS (FFT)
    # ============================================================
    
    def check_frequency():
        try:
            # Perform FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Log scale for better visualization
            magnitude_log = np.log1p(magnitude)
            
            # High frequency energy indicates tampering
            h, w = magnitude_log.shape
            
            # Get center (low frequency) and edges (high frequency)
            center = magnitude_log[h//4:3*h//4, w//4:3*w//4]
            
            center_energy = np.sum(center)
            total_energy = np.sum(magnitude_log)
            
            # High frequency ratio
            high_freq_ratio = (total_energy - center_energy) / (total_energy + 1)
            
            # Tampering often increases high frequency components
            frequency_score = high_freq_ratio * 100
            
            return min(frequency_score, 100)
        except:
            return 0
    
    scores['frequency'] = check_frequency()
    
    
    # ============================================================
    # METHOD 5: EDGE DETECTION AND CONSISTENCY
    # ============================================================
    
    def check_edges():
        try:
            # Canny edge detection with lower thresholds (more sensitive)
            edges = cv2.Canny(gray, 30, 100)  # More aggressive than 50, 150
            
            # Find contours
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1]) * 100
            
            # Count edges in different regions
            h, w = edges.shape
            regions_edge_density = []
            
            for i in range(3):
                for j in range(3):
                    region = edges[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
                    density = np.sum(region) / (region.shape[0] * region.shape[1]) * 100
                    regions_edge_density.append(density)
            
            # High variation in edge density = suspicious
            avg_density = np.mean(regions_edge_density)
            std_density = np.std(regions_edge_density)
            
            if avg_density > 0:
                edge_score = (std_density / avg_density) * 60
            else:
                edge_score = 0
            
            return min(edge_score, 100)
        except:
            return 0
    
    scores['edges'] = check_edges()
    
    
    # ============================================================
    # METHOD 6: BANDING AND POSTERIZATION DETECTION
    # ============================================================
    
    def check_banding():
        try:
            # Detect banding (common in tampered documents)
            # Create a high-pass filter to detect banding
            kernel = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]])
            
            # Apply to grayscale
            high_pass = cv2.filter2D(gray, -1, kernel)
            
            # Calculate variance of high-pass filtered image
            # High variance = banding detected
            high_pass_var = np.var(high_pass)
            
            # Also check for color banding in RGB
            b, g, r = cv2.split(img)
            
            # Calculate banding in each channel
            r_hp = cv2.filter2D(r, -1, kernel)
            g_hp = cv2.filter2D(g, -1, kernel)
            b_hp = cv2.filter2D(b, -1, kernel)
            
            channel_vars = [np.var(r_hp), np.var(g_hp), np.var(b_hp)]
            avg_channel_var = np.mean(channel_vars)
            
            # Normalize to 0-100
            banding_score = min(high_pass_var / 50, 100)
            
            return banding_score
        except:
            return 0
    
    scores['banding'] = check_banding()
    
    
    # ============================================================
    # CALCULATE FINAL SCORE WITH NEW WEIGHTS
    # ============================================================
    
    # New weights: more emphasis on newly added methods
    weights = {
        'compression': 0.25,
        'blur': 0.20,
        'histogram': 0.15,
        'frequency': 0.15,
        'edges': 0.15,
        'banding': 0.10
    }
    
    # Calculate weighted score
    final_score = 0
    for method, weight in weights.items():
        final_score += scores.get(method, 0) * weight
    
    final_score = round(final_score, 2)
    
    
    # ============================================================
    # DETERMINE STATUS WITH ADJUSTED THRESHOLDS
    # ============================================================
    
    # Lower thresholds for better detection
    if final_score < 25:
        status = "LOW"
        message = "✅ No significant tampering detected."
    elif final_score < 55:
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
            "histogram_anomalies": round(scores['histogram'], 2),
            "frequency_domain": round(scores['frequency'], 2),
            "edge_discontinuities": round(scores['edges'], 2),
            "banding_detection": round(scores['banding'], 2)
        },
        "weights": weights
    }


if __name__ == "__main__":
    test_image_path = "test_image.png"
    
    try:
        result = detect_tampering(test_image_path)
        
        print("=" * 60)
        print("TAMPERING DETECTION RESULTS (V2)")
        print("=" * 60)
        print(f"Status: {result['status']}")
        print(f"Score: {result['score']}/100")
        print(f"Message: {result['message']}")
        print(f"\nDetailed Analysis:")
        for method, score in result['detailed_scores'].items():
            print(f"  - {method}: {score}")
        print("=" * 60)
    
    except Exception as e:
        print(f"Error: {e}")
