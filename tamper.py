from PIL import Image, ImageFilter, ImageStat
import io


def detect_tampering(document_image):
    """
    Prototype document tampering detection module.

    This module looks for basic image-level indicators that may
    be useful for detecting possible manipulation.

    It does NOT prove that a document is fake or tampered.
    The result is intended for prototype demonstration only.
    """

    try:
        # ----------------------------------------------------
        # BASIC IMAGE INFORMATION
        # ----------------------------------------------------

        width, height = document_image.size

        image_format = getattr(
            document_image,
            "format",
            "Unknown"
        )

        color_mode = document_image.mode

        # ----------------------------------------------------
        # IMAGE QUALITY ANALYSIS
        # ----------------------------------------------------

        gray_image = document_image.convert("L")

        statistics = ImageStat.Stat(gray_image)

        mean_brightness = statistics.mean[0]

        # Standard deviation indicates how much variation
        # exists in the image.
        contrast = statistics.stddev[0]

        # ----------------------------------------------------
        # EDGE / SHARPNESS ANALYSIS
        # ----------------------------------------------------

        edge_image = gray_image.filter(
            ImageFilter.FIND_EDGES
        )

        edge_statistics = ImageStat.Stat(edge_image)

        edge_strength = edge_statistics.mean[0]

        # ----------------------------------------------------
        # IMAGE SIZE CHECK
        # ----------------------------------------------------

        pixel_count = width * height

        score = 0

        indicators = []

        # ----------------------------------------------------
        # INDICATOR 1: VERY SMALL IMAGE
        # ----------------------------------------------------

        if pixel_count < 300000:

            score += 15

            indicators.append(
                "Low image resolution detected."
            )

        else:

            indicators.append(
                "Image resolution is adequate."
            )

        # ----------------------------------------------------
        # INDICATOR 2: EXTREMELY LOW CONTRAST
        # ----------------------------------------------------

        if contrast < 20:

            score += 15

            indicators.append(
                "Unusually low image contrast detected."
            )

        else:

            indicators.append(
                "Image contrast appears normal."
            )

        # ----------------------------------------------------
        # INDICATOR 3: EXTREMELY HIGH CONTRAST
        # ----------------------------------------------------

        if contrast > 100:

            score += 10

            indicators.append(
                "Unusually high image contrast detected."
            )

        # ----------------------------------------------------
        # INDICATOR 4: EDGE PATTERN
        # ----------------------------------------------------

        if edge_strength < 2:

            score += 10

            indicators.append(
                "Very low edge activity detected."
            )

        elif edge_strength > 35:

            score += 10

            indicators.append(
                "High edge activity detected."
            )

        else:

            indicators.append(
                "Edge characteristics appear normal."
            )

        # ----------------------------------------------------
        # INDICATOR 5: EXTREME BRIGHTNESS
        # ----------------------------------------------------

        if mean_brightness < 20:

            score += 10

            indicators.append(
                "Image appears unusually dark."
            )

        elif mean_brightness > 240:

            score += 10

            indicators.append(
                "Image appears unusually bright."
            )

        # ----------------------------------------------------
        # LIMIT SCORE
        # ----------------------------------------------------

        score = min(score, 100)

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        if score < 30:

            status = "LOW SUSPICION"

            message = (
                "No strong image-level manipulation indicators "
                "were detected by the prototype analysis."
            )

        elif score < 60:

            status = "MEDIUM SUSPICION"

            message = (
                "Some unusual image characteristics were detected. "
                "Further forensic analysis is recommended."
            )

        else:

            status = "HIGH SUSPICION"

            message = (
                "Multiple unusual image characteristics were detected. "
                "The document should undergo further verification."
            )

        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return {

            "status": status,

            "score": score,

            "message": message,

            "indicators": indicators,

            "image_width": width,

            "image_height": height,

            "image_format": image_format,

            "color_mode": color_mode,

            "brightness": round(mean_brightness, 2),

            "contrast": round(contrast, 2),

            "edge_strength": round(edge_strength, 2)
        }

    except Exception as error:

        return {

            "status": "ERROR",

            "score": 0,

            "message": (
                "Tampering analysis could not be completed."
            ),

            "indicators": [
                str(error)
            ]
        }