TAMPERING DETECTION MODULE V4
For SIH26188 - AI-Powered Fake Identity & Document Screening

V4:
- Localized sharpness analysis
- Localized noise/residual analysis
- Localized edge analysis
- Localized color consistency
- Localized texture analysis
- Frequency analysis
- Compression supporting signal
- Robust anomaly scoring

IMPORTANT:
This is a prototype screening system.
The score is an anomaly/suspicion score and is NOT proof
that a document is fake or authentic.
"""

import cv2
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# IMAGE LOADING
# ============================================================

def _load_image(document_image):
    """
    Convert PIL image, NumPy image, or file path
    into an OpenCV BGR image.
    """

    if isinstance(document_image, Image.Image):

        image = np.array(document_image)

        if image.ndim == 2:
            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        if image.shape[2] == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR
            )

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

    if isinstance(document_image, np.ndarray):

        image = document_image.copy()

        if image.ndim == 2:
            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        if image.shape[2] == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR
            )

        return image

    image = cv2.imread(str(document_image))

    return image


# ============================================================
# ROBUST ANOMALY SCORING
# ============================================================

def _normalize_scores(values):
    """
    Convert local measurements into anomaly scores
    using median/MAD statistics.
    """

    values = np.asarray(
        values,
        dtype=np.float32
    )

    if len(values) == 0:
        return np.array([])

    median = np.median(values)

    mad = np.median(
        np.abs(values - median)
    )

    scale = max(
        mad * 1.4826,
        1e-6
    )

    z = np.abs(values - median) / scale

    scores = np.clip(
        (z / 4.0) * 100.0,
        0,
        100
    )

    return scores


# ============================================================
# GRID CREATION
# ============================================================

def _grid_regions(image, rows=4, cols=4):
    """
    Divide image into 4x4 local regions.
    """

    h, w = image.shape[:2]

    regions = []

    for r in range(rows):

        y1 = int(
            r * h / rows
        )

        y2 = int(
            (r + 1) * h / rows
        )

        for c in range(cols):

            x1 = int(
                c * w / cols
            )

            x2 = int(
                (c + 1) * w / cols
            )

            region = image[
                y1:y2,
                x1:x2
            ]

            if region.size > 0:
                regions.append(region)

    return regions


# ============================================================
# 1. LOCAL SHARPNESS
# ============================================================

def _check_local_sharpness(gray):

    try:

        regions = _grid_regions(gray)

        values = []

        for region in regions:

            if (
                region.shape[0] < 10
                or region.shape[1] < 10
            ):
                continue

            laplacian = cv2.Laplacian(
                region,
                cv2.CV_64F
            )

            sharpness = float(
                np.var(laplacian)
            )

            values.append(sharpness)

        if len(values) < 4:
            return 0.0

        anomaly_scores = _normalize_scores(values)

        suspicious = anomaly_scores[
            anomaly_scores > 35
        ]

        if len(suspicious) == 0:
            return 0.0

        score = float(
            np.mean(suspicious)
        )

        ratio = (
            len(suspicious)
            / len(anomaly_scores)
        )

        score *= min(
            1.0,
            ratio * 4.0 + 0.2
        )

        return float(
            np.clip(score, 0, 100)
        )

    except Exception:
        return 0.0


# ============================================================
# 2. LOCAL NOISE / RESIDUAL
# ============================================================

def _check_local_noise(gray):

    try:

        smooth = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        residual = cv2.absdiff(
            gray,
            smooth
        )

        regions = _grid_regions(
            residual
        )

        values = []

        for region in regions:

            if region.size == 0:
                continue

            noise_level = float(
                np.std(region)
            )

            values.append(noise_level)

        if len(values) < 4:
            return 0.0

        anomaly_scores = _normalize_scores(values)

        suspicious = anomaly_scores[
            anomaly_scores > 35
        ]

        if len(suspicious) == 0:
            return 0.0

        score = float(
            np.mean(suspicious)
        )

        ratio = (
            len(suspicious)
            / len(anomaly_scores)
        )

        score *= min(
            1.0,
            ratio * 4.0 + 0.2
        )

        return float(
            np.clip(score, 0, 100)
        )

    except Exception:
        return 0.0


# ============================================================
# 3. LOCAL EDGE ANALYSIS
# ============================================================

def _check_local_edges(gray):

    try:

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        regions = _grid_regions(
            edges
        )

        values = []

        for region in regions:

            if region.size == 0:
                continue

            density = float(
                np.mean(region > 0) * 100
            )

            values.append(density)

        if len(values) < 4:
            return 0.0

        anomaly_scores = _normalize_scores(values)

        suspicious = anomaly_scores[
            anomaly_scores > 35
        ]

        if len(suspicious) == 0:
            return 0.0

        score = float(
            np.mean(suspicious)
        )

        ratio = (
            len(suspicious)
            / len(anomaly_scores)
        )

        score *= min(
            1.0,
            ratio * 4.0 + 0.2
        )

        return float(
            np.clip(score, 0, 100)
        )

    except Exception:
        return 0.0


# ============================================================
# 4. LOCAL COLOR ANALYSIS
# ============================================================

def _check_color_consistency(img):

    try:

        hsv = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2HSV
        )

        h_regions = _grid_regions(
            hsv[:, :, 0]
        )

        s_regions = _grid_regions(
            hsv[:, :, 1]
        )

        v_regions = _grid_regions(
            hsv[:, :, 2]
        )

        values = []

        for (
            h_region,
            s_region,
            v_region
        ) in zip(
            h_regions,
            s_regions,
            v_regions
        ):

            h_mean = float(
                np.mean(h_region)
            )

            s_mean = float(
                np.mean(s_region)
            )

            v_mean = float(
                np.mean(v_region)
            )

            h_norm = h_mean / 180.0
            s_norm = s_mean / 255.0
            v_norm = v_mean / 255.0

            value = (
                h_norm * 0.30
                + s_norm * 0.35
                + v_norm * 0.35
            )

            values.append(value)

        if len(values) < 4:
            return 0.0

        anomaly_scores = _normalize_scores(values)

        suspicious = anomaly_scores[
            anomaly_scores > 40
        ]

        if len(suspicious) == 0:
            return 0.0

        score = float(
            np.mean(suspicious)
        )

        ratio = (
            len(suspicious)
            / len(anomaly_scores)
        )

        score *= min(
            1.0,
            ratio * 4.0 + 0.15
        )

        return float(
            np.clip(score, 0, 100)
        )

    except Exception:
        return 0.0


# ============================================================
# 5. LOCAL TEXTURE ANALYSIS
# ============================================================

def _check_texture(gray):

    try:

        regions = _grid_regions(
            gray
        )

        values = []

        for region in regions:

            if region.size == 0:
                continue

            texture = float(
                np.std(region)
            )

            values.append(texture)

        if len(values) < 4:
            return 0.0

        anomaly_scores = _normalize_scores(values)

        suspicious = anomaly_scores[
            anomaly_scores > 40
        ]

        if len(suspicious) == 0:
            return 0.0

        score = float(
            np.mean(suspicious)
        )

        ratio = (
            len(suspicious)
            / len(anomaly_scores)
        )

        score *= min(
            1.0,
            ratio * 4.0 + 0.15
        )

        return float(
            np.clip(score, 0, 100)
        )

    except Exception:
        return 0.0


# ============================================================
# 6. FREQUENCY ANALYSIS
# ============================================================

def _check_frequency(gray):

    try:

        image_float = np.float32(
            gray
        )

        fft = np.fft.fft2(
            image_float
        )

        fft_shift = np.fft.fftshift(
            fft
        )

        magnitude = np.abs(
            fft_shift
        )

        magnitude_log = np.log1p(
            magnitude
        )

        h, w = magnitude_log.shape

        cy = h // 2
        cx = w // 2

        radius = max(
            min(h, w) // 8,
            1
        )

        y1 = max(
            0,
            cy - radius
        )

        y2 = min(
            h,
            cy + radius
        )

        x1 = max(
            0,
            cx - radius
        )

        x2 = min(
            w,
            cx + radius
        )

        low_frequency = magnitude_log[
            y1:y2,
            x1:x2
        ]

        total_energy = float(
            np.sum(magnitude_log)
        )

        low_energy = float(
            np.sum(low_frequency)
        )

        if total_energy <= 0:
            return 0.0

        high_frequency_ratio = (
            total_energy - low_energy
        ) / total_energy

        score = abs(
            high_frequency_ratio - 0.55
        ) * 120

        return float(
            np.clip(score, 0, 100)
        )

    except Exception:
        return 0.0


# ============================================================
# 7. COMPRESSION SUPPORTING SIGNAL
# ============================================================

def _check_compression(gray):

    try:

        h, w = gray.shape

        block_size = 8

        values = []

        for y in range(
            0,
            h - block_size + 1,
            block_size
        ):

            for x in range(
                0,
                w - block_size + 1,
                block_size
            ):

                block = gray[
                    y:y + block_size,
                    x:x + block_size
                ]

                if block.size == 0:
                    continue

                values.append(
                    float(
                        np.var(block)
                    )
                )

        if len(values) < 8:
            return 0.0

        values = np.asarray(
            values,
            dtype=np.float32
        )

        median = np.median(values)

        mad = np.median(
            np.abs(values - median)
        )

        if mad < 1e-6:
            return 0.0

        robust_z = (
            np.abs(values - median)
            / (mad * 1.4826 + 1e-6)
        )

        suspicious_ratio = float(
            np.mean(
                robust_z > 4
            )
        )

        score = (
            suspicious_ratio * 100
        )

        return float(
            np.clip(score, 0, 100)
        )

    except Exception:
        return 0.0


# ============================================================
# FINAL SCORE
# ============================================================

def _calculate_final_score(scores):

    weights = {

        "local_sharpness": 0.25,

        "local_noise": 0.25,

        "local_edges": 0.18,

        "local_color": 0.12,

        "local_texture": 0.10,

        "frequency": 0.06,

        "compression": 0.04
    }

    final_score = 0.0

    for method, weight in weights.items():

        final_score += (
            scores.get(
                method,
                0.0
            ) * weight
        )

    final_score = np.clip(
        final_score,
        0,
        100
    )

    return round(
        float(final_score),
        2
    ), weights


# ============================================================
# MAIN FUNCTION
# ============================================================

def detect_tampering(document_image):

    try:

        # Load image
        img = _load_image(
            document_image
        )

        if img is None:

            return {
                "status": "ERROR",
                "score": 0,
                "message": (
                    "Could not read "
                    "the document image."
                ),
                "detailed_scores": {},
                "weights": {}
            }

        if img.size == 0:

            return {
                "status": "ERROR",
                "score": 0,
                "message": (
                    "Document image "
                    "is empty."
                ),
                "detailed_scores": {},
                "weights": {}
            }

        # Resize very large images
        max_dimension = 1800

        h, w = img.shape[:2]

        if max(h, w) > max_dimension:

            scale = (
                max_dimension
                / max(h, w)
            )

            new_w = max(
                1,
                int(w * scale)
            )

            new_h = max(
                1,
                int(h * scale)
            )

            img = cv2.resize(
                img,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

        # Convert to grayscale
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # Calculate signals
        scores = {}

        scores["local_sharpness"] = (
            _check_local_sharpness(
                gray
            )
        )

        scores["local_noise"] = (
            _check_local_noise(
                gray
            )
        )

        scores["local_edges"] = (
            _check_local_edges(
                gray
            )
        )

        scores["local_color"] = (
            _check_color_consistency(
                img
            )
        )

        scores["local_texture"] = (
            _check_texture(
                gray
            )
        )

        scores["frequency"] = (
            _check_frequency(
                gray
            )
        )

        scores["compression"] = (
            _check_compression(
                gray
            )
        )

        # Calculate final score
        final_score, weights = (
            _calculate_final_score(
                scores
            )
        )

        # Classification
        if final_score < 20:

            status = "LOW"

            message = (
                "No significant image-level "
                "anomalies were identified "
                "during preliminary screening."
            )

        elif final_score < 40:

            status = "MEDIUM"

            message = (
                "Some localized image anomalies "
                "were identified. Manual review "
                "is recommended."
            )

        else:

            status = "HIGH"

            message = (
                "Multiple image-level anomalies "
                "were identified. Further "
                "verification is recommended."
            )

        # Detailed results
        detailed_scores = {

            "local_sharpness": round(
                scores[
                    "local_sharpness"
                ],
                2
            ),

            "local_noise": round(
                scores[
                    "local_noise"
                ],
                2
            ),

            "local_edge_anomalies": round(
                scores[
                    "local_edges"
                ],
                2
            ),

            "local_color_anomalies": round(
                scores[
                    "local_color"
                ],
                2
            ),

            "local_texture_anomalies": round(
                scores[
                    "local_texture"
                ],
                2
            ),

            "frequency_analysis": round(
                scores[
                    "frequency"
                ],
                2
            ),

            "compression_analysis": round(
                scores[
                    "compression"
                ],
                2
            )
        }

        # Return result
        return {

            "status": status,

            "score": final_score,

            "message": message,

            "detailed_scores": (
                detailed_scores
            ),

            "weights": weights
        }

    except Exception as e:

        return {

            "status": "ERROR",

            "score": 0,

            "message": (
                "Tampering analysis "
                "could not be completed."
            ),

            "detailed_scores": {},

            "weights": {},

            "error": str(e)
        }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    test_image_path = (
        "test_image.png"
    )

    print("=" * 65)

    print(
        "TAMPERING DETECTION MODULE V4"
    )

    print("=" * 65)

    try:

        result = detect_tampering(
            test_image_path
        )

        print(
            f"Status : "
            f"{result.get('status')}"
        )

        print(
            f"Score  : "
            f"{result.get('score')}/100"
        )

        print(
            f"Message: "
            f"{result.get('message')}"
        )

        print()

        print(
            "Detailed Analysis:"
        )

        print("-" * 65)

        for (
            method,
            score
        ) in result.get(
            "detailed_scores",
            {}
        ).items():

            print(
                f"{method:35s}: "
                f"{score}"
            )

        print("=" * 65)

    except Exception as e:

        print(
            f"Error while testing V4: {e}"
        )
