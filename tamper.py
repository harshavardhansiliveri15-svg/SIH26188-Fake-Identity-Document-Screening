```python
"""
TAMPERING DETECTION MODULE V4
For SIH26188 - AI-Powered Fake Identity & Document Screening

V4 improvements:
- Localized sharpness inconsistency
- Localized noise/residual inconsistency
- Localized edge-density anomalies
- Localized color anomalies
- Block-level analysis instead of relying mainly on global image statistics
- Robust scoring to reduce false positives from normal document layouts
- Compatible with existing app.py:
      detect_tampering(document_image)

IMPORTANT:
This is a prototype screening module.
The score is a suspicion/anomaly score, NOT proof that a document is fake.
"""

import cv2
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# Utility functions
# ============================================================

def _load_image(document_image):
    """
    Convert PIL image / NumPy image / file path into OpenCV BGR image.
    """

    if isinstance(document_image, Image.Image):
        image = np.array(document_image)

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if isinstance(document_image, np.ndarray):

        image = document_image.copy()

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        return image

    image = cv2.imread(str(document_image))

    return image


def _normalize_scores(values):
    """
    Convert an array of values into anomaly scores using
    robust median/MAD statistics.

    This is better than comparing everything against fixed
    thresholds because normal documents can have very different
    layouts, colors and text densities.
    """

    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return np.array([])

    median = np.median(values)

    mad = np.median(np.abs(values - median))

    # Prevent division by zero on very uniform images
    scale = max(mad * 1.4826, 1e-6)

    z = np.abs(values - median) / scale

    # Convert robust z-score to 0-100
    scores = np.clip((z / 4.0) * 100.0, 0, 100)

    return scores


def _grid_regions(gray, rows=4, cols=4):
    """
    Split image into regular blocks.
    """

    h, w = gray.shape

    regions = []

    for r in range(rows):
        y1 = int(r * h / rows)
        y2 = int((r + 1) * h / rows)

        for c in range(cols):
            x1 = int(c * w / cols)
            x2 = int((c + 1) * w / cols)

            region = gray[y1:y2, x1:x2]

            if region.size > 0:
                regions.append(region)

    return regions


# ============================================================
# 1. Local Sharpness Analysis
# ============================================================

def _check_local_sharpness(gray):
    """
    Detect blocks whose sharpness is unusually different from
    the rest of the document.

    Useful as a supporting signal when a region has been replaced,
    blurred, pasted or resampled.
    """

    try:
        regions = _grid_regions(gray)

        sharpness_values = []

        for region in regions:

            if region.shape[0] < 10 or region.shape[1] < 10:
                continue

            lap = cv2.Laplacian(region, cv2.CV_64F)

            variance = float(np.var(lap))

            sharpness_values.append(variance)

        if len(sharpness_values) < 4:
            return 0.0

        anomaly_scores = _normalize_scores(sharpness_values)

        # Only count stronger local anomalies
        suspicious = anomaly_scores[anomaly_scores > 35]

        if len(suspicious) == 0:
            return 0.0

        score = float(np.mean(suspicious))

        # Small number of suspicious blocks should not dominate
        ratio = len(suspicious) / len(anomaly_scores)

        score *= min(1.0, ratio * 4.0 + 0.2)

        return float(np.clip(score, 0, 100))

    except Exception:
        return 0.0


# ============================================================
# 2. Local Noise / Residual Analysis
# ============================================================

def _check_local_noise(gray):
    """
    Analyze high-frequency residual noise in local regions.

    Edited/replaced areas can sometimes have a different noise
    pattern from surrounding areas.
    """

    try:

        # Estimate smooth image
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)

        # High-frequency residual
        residual = cv2.absdiff(gray, smooth)

        regions = _grid_regions(residual)

        noise_values = []

        for region in regions:

            if region.size == 0:
                continue

            value = float(np.std(region))

            noise_values.append(value)

        if len(noise_values) < 4:
            return 0.0

        anomaly_scores = _normalize_scores(noise_values)

        suspicious = anomaly_scores[anomaly_scores > 35]

        if len(suspicious) == 0:
            return 0.0

        score = float(np.mean(suspicious))

        ratio = len(suspicious) / len(anomaly_scores)

        score *= min(1.0, ratio * 4.0 + 0.2)

        return float(np.clip(score, 0, 100))

    except Exception:
        return 0.0


# ============================================================
# 3. Local Edge Analysis
# ============================================================

def _check_local_edges(gray):
    """
    Detect unusually different edge density between regions.

    This can identify areas whose text/graphics structure differs
    significantly from surrounding document regions.
    """

    try:

        edges = cv2.Canny(gray, 50, 150)

        regions = _grid_regions(edges)

        edge_values = []

        for region in regions:

            if region.size == 0:
                continue

            density = float(np.mean(region > 0) * 100)

            edge_values.append(density)

        if len(edge_values) < 4:
            return 0.0

        anomaly_scores = _normalize_scores(edge_values)

        suspicious = anomaly_scores[anomaly_scores > 35]

        if len(suspicious) == 0:
            return 0.0

        score = float(np.mean(suspicious))

        ratio = len(suspicious) / len(anomaly_scores)

        score *= min(1.0, ratio * 4.0 + 0.2)

        return float(np.clip(score, 0, 100))

    except Exception:
        return 0.0


# ============================================================
# 4. Local Color Consistency
# ============================================================

def _check_color_consistency(img):
    """
    Analyze local color statistics.

    Uses HSV channels instead of simply comparing the entire
    image's average color.

    A document-wide red/blue/green background should not by itself
    be considered tampering.
    """

    try:

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_regions = _grid_regions(hsv[:, :, 0])
        s_regions = _grid_regions(hsv[:, :, 1])
        v_regions = _grid_regions(hsv[:, :, 2])

        color_values = []

        for h_region, s_region, v_region in zip(
            h_regions,
            s_regions,
            v_regions
        ):

            h_mean = float(np.mean(h_region))
            s_mean = float(np.mean(s_region))
            v_mean = float(np.mean(v_region))

            # Normalize the three channels
            h_norm = h_mean / 180.0
            s_norm = s_mean / 255.0
            v_norm = v_mean / 255.0

            value = (
                h_norm * 0.30 +
                s_norm * 0.35 +
                v_norm * 0.35
            )

            color_values.append(value)

        if len(color_values) < 4:
            return 0.0

        anomaly_scores = _normalize_scores(color_values)

        suspicious = anomaly_scores[anomaly_scores > 40]

        if len(suspicious) == 0:
            return 0.0

        score = float(np.mean(suspicious))

        ratio = len(suspicious) / len(anomaly_scores)

        score *= min(1.0, ratio * 4.0 + 0.15)

        return float(np.clip(score, 0, 100))

    except Exception:
        return 0.0


# ============================================================
# 5. Local Texture Analysis
# ============================================================

def _check_texture(gray):
    """
    Compare local texture/contrast between document regions.

    This is useful for detecting a region that has a noticeably
    different texture from the rest of the document.
    """

    try:

        regions = _grid_regions(gray)

        texture_values = []

        for region in regions:

            if region.size == 0:
                continue

            value = float(np.std(region))

            texture_values.append(value)

        if len(texture_values) < 4:
            return 0.0

        anomaly_scores = _normalize_scores(texture_values)

        suspicious = anomaly_scores[anomaly_scores > 40]

        if len(suspicious) == 0:
            return 0.0

        score = float(np.mean(suspicious))

        ratio = len(suspicious) / len(anomaly_scores)

        score *= min(1.0, ratio * 4.0 + 0.15)

        return float(np.clip(score, 0, 100))

    except Exception:
        return 0.0


# ============================================================
# 6. Frequency Analysis
# ============================================================

def _check_frequency(gray):
    """
    Global frequency analysis.

    This is deliberately given a low weight because frequency
    statistics alone cannot reliably prove image manipulation.
    """

    try:

        image_float = np.float32(gray)

        fft = np.fft.fft2(image_float)

        fft_shift = np.fft.fftshift(fft)

        magnitude = np.abs(fft_shift)

        magnitude_log = np.log1p(magnitude)

        h, w = magnitude_log.shape

        cy = h // 2
        cx = w // 2

        radius = min(h, w) // 8

        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius)

        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius)

        low_frequency = magnitude_log[y1:y2, x1:x2]

        total_energy = float(np.sum(magnitude_log))

        low_energy = float(np.sum(low_frequency))

        if total_energy <= 0:
            return 0.0

        high_frequency_ratio = (
            (total_energy - low_energy) /
            total_energy
        )

        # Map typical ratios into a conservative score.
        score = abs(high_frequency_ratio - 0.55) * 120

        return float(np.clip(score, 0, 100))

    except Exception:
        return 0.0


# ============================================================
# 7. JPEG / Compression Supporting Signal
# ============================================================

def _check_compression(gray):
    """
    Supporting compression analysis.

    This is intentionally low-weight because compression artifacts
    can come from normal image saving/resizing.
    """

    try:

        h, w = gray.shape

        block_variances = []

        block_size = 8

        for y in range(0, h - block_size + 1, block_size):

            for x in range(0, w - block_size + 1, block_size):

                block = gray[
                    y:y + block_size,
                    x:x + block_size
                ]

                if block.size == 0:
                    continue

                block_variances.append(float(np.var(block)))

        if len(block_variances) < 8:
            return 0.0

        values = np.asarray(block_variances)

        median = np.median(values)

        mad = np.median(np.abs(values - median))

        if mad < 1e-6:
            return 0.0

        robust_z = np.abs(values - median) / (
            mad * 1.4826 + 1e-6
        )

        suspicious_ratio = np.mean(robust_z > 4)

        score = suspicious_ratio * 100

        return float(np.clip(score, 0, 100))

    except Exception:
        return 0.0


# ============================================================
# 8. Combine Scores
# ============================================================

def _calculate_final_score(scores):
    """
    Combine all signals.

    Local signals receive most of the weight.
    Global signals receive smaller weights.
    """

    weights = {

        # Main local signals
        "local_sharpness": 0.25,
        "local_noise": 0.25,
        "local_edges": 0.18,
        "local_color": 0.12,
        "local_texture": 0.10,

        # Supporting global signals
        "frequency": 0.06,
        "compression": 0.04
    }

    final_score = 0.0

    for method, weight in weights.items():

        final_score += (
            scores.get(method, 0.0) *
            weight
        )

    return round(
        float(np.clip(final_score, 0, 100)),
        2
    ), weights


# ============================================================
# Main Tampering Detection Function
# ============================================================

def detect_tampering(document_image):

    try:

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        img = _load_image(document_image)

        if img is None:

            return {
                "status": "ERROR",
                "score": 0,
                "message": "Could not read the document image.",
                "detailed_scores": {},
                "weights": {}
            }

        # ----------------------------------------------------
        # Basic image validation
        # ----------------------------------------------------

        if img.size == 0:

            return {
                "status": "ERROR",
                "score": 0,
                "message": "Document image is empty.",
                "detailed_scores": {},
                "weights": {}
            }

        # ----------------------------------------------------
        # Resize extremely large images for stable processing
        # ----------------------------------------------------

        max_dimension = 1800

        h, w = img.shape[:2]

        if max(h, w) > max_dimension:

            scale = max_dimension / max(h, w)

            new_w = int(w * scale)
            new_h = int(h * scale)

            img = cv2.resize(
                img,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Calculate individual signals
        # ----------------------------------------------------

        scores = {}

        scores["local_sharpness"] = (
            _check_local_sharpness(gray)
        )

        scores["local_noise"] = (
            _check_local_noise(gray)
        )

        scores["local_edges"] = (
            _check_local_edges(gray)
        )

        scores["local_color"] = (
            _check_color_consistency(img)
        )

        scores["local_texture"] = (
            _check_texture(gray)
        )

        scores["frequency"] = (
            _check_frequency(gray)
        )

        scores["compression"] = (
            _check_compression(gray)
        )

        # ----------------------------------------------------
        # Calculate final score
        # ----------------------------------------------------

        final_score, weights = _calculate_final_score(
            scores
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        if final_score < 20:

            status = "LOW"

            message = (
                "No significant image-level anomalies "
                "were identified during preliminary screening."
            )

        elif final_score < 40:

            status = "MEDIUM"

            message = (
                "Some localized image anomalies were identified. "
                "Manual review is recommended."
            )

        else:

            status = "HIGH"

            message = (
                "Multiple image-level anomalies were identified. "
                "Further verification is recommended."
            )

        # ----------------------------------------------------
        # Detailed scores
        # ----------------------------------------------------

        detailed_scores = {

            "local_sharpness": round(
                scores["local_sharpness"],
                2
            ),

            "local_noise": round(
                scores["local_noise"],
                2
            ),

            "local_edge_anomalies": round(
                scores["local_edges"],
                2
            ),

            "local_color_anomalies": round(
                scores["local_color"],
                2
            ),

            "local_texture_anomalies": round(
                scores["local_texture"],
                2
            ),

            "frequency_analysis": round(
                scores["frequency"],
                2
            ),

            "compression_analysis": round(
                scores["compression"],
                2
            )
        }

        # ----------------------------------------------------
        # Return result compatible with existing dashboard
        # ----------------------------------------------------

        return {

            "status": status,

            "score": final_score,

            "message": message,

            "detailed_scores": detailed_scores,

            "weights": weights

        }

    except Exception as e:

        return {

            "status": "ERROR",

            "score": 0,

            "message": (
                "Tampering analysis could not be completed."
            ),

            "detailed_scores": {},

            "weights": {},

            "error": str(e)

        }


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    test_image_path = "test_image.png"

    print("=" * 65)
    print("TAMPERING DETECTION MODULE V4")
    print("=" * 65)

    try:

        result = detect_tampering(
            test_image_path
        )

        print(
            f"Status : {result.get('status')}"
        )

        print(
            f"Score  : {result.get('score')}/100"
        )

        print(
            f"Message: {result.get('message')}"
        )

        print()
        print("Detailed Analysis:")
        print("-" * 65)

        for method, score in result.get(
            "detailed_scores",
            {}
        ).items():

            print(
                f"{method:35s}: {score}"
            )

        print("=" * 65)

    except Exception as e:

        print(
            f"Error while testing V4: {e}"
        )
```
