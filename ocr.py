import re
import cv2
import numpy as np
from PIL import Image
from rapidocr import RapidOCR


# ---------------------------------------------------------
# OCR ENGINE
# ---------------------------------------------------------

_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine

    if _ocr_engine is None:
        _ocr_engine = RapidOCR()

    return _ocr_engine


# ---------------------------------------------------------
# IMAGE CONVERSION
# ---------------------------------------------------------

def convert_to_bgr(image):
    """
    Convert uploaded image/PIL image/numpy image to BGR format.
    """

    if isinstance(image, Image.Image):
        image = np.array(image)

    if not isinstance(image, np.ndarray):
        raise ValueError("Unsupported image format")

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------

def preprocess_image(image):
    """
    Improve image quality before OCR.
    """

    image = convert_to_bgr(image)

    # Enlarge small document images
    height, width = image.shape[:2]

    if width < 1800:
        scale = 1800 / width
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Convert back to BGR because OCR engines generally
    # handle standard image arrays better
    processed = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2BGR
    )

    return processed


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def normalize_text(text):
    if not text:
        return ""

    text = str(text)

    # Replace repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_lines(lines):
    cleaned = []

    for line in lines:
        line = normalize_text(line)

        if line and len(line) >= 2:
            cleaned.append(line)

    return cleaned


# ---------------------------------------------------------
# DOCUMENT TYPE DETECTION
# ---------------------------------------------------------

def detect_document_type(text):
    text_upper = text.upper()

    if "INCOME TAX DEPARTMENT" in text_upper:
        if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text_upper):
            return "PAN Card"

    if "PERMANENT ACCOUNT NUMBER" in text_upper:
        return "PAN Card"

    if "GOVERNMENT OF INDIA" in text_upper:
        if "AADHAAR" in text_upper or "UNIQUE IDENTIFICATION" in text_upper:
            return "Aadhaar Card"

    if "AADHAAR" in text_upper:
        return "Aadhaar Card"

    if "ELECTION COMMISSION" in text_upper:
        return "Voter ID"

    if "DRIVING LICENCE" in text_upper or "DRIVING LICENSE" in text_upper:
        return "Driving Licence"

    if "PASSPORT" in text_upper:
        return "Passport"

    return "Unknown"


# ---------------------------------------------------------
# PAN NUMBER EXTRACTION
# ---------------------------------------------------------

def extract_pan_number(text):
    """
    PAN format:
    AAAAA9999A
    """

    text = text.upper()

    pattern = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"

    match = re.search(pattern, text)

    if match:
        return match.group(0)

    # Try removing spaces caused by OCR
    compact = re.sub(r"[^A-Z0-9]", "", text)

    match = re.search(
        r"[A-Z]{5}[0-9]{4}[A-Z]",
        compact
    )

    if match:
        return match.group(0)

    return ""


# ---------------------------------------------------------
# DATE EXTRACTION
# ---------------------------------------------------------

def extract_date(text):
    """
    Detect common DOB/date formats.
    """

    patterns = [
        r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
        r"\b\d{2}[/-]\d{2}[/-]\d{2}\b",
        r"\b\d{4}[/-]\d{2}[/-]\d{2}\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(0)

    return ""


# ---------------------------------------------------------
# NAME EXTRACTION
# ---------------------------------------------------------

def extract_name(lines):
    """
    Try to identify a person's name from OCR lines.
    """

    keywords = [
        "NAME",
        "FULL NAME",
        "SURNAME",
        "GIVEN NAME"
    ]

    for i, line in enumerate(lines):

        upper_line = line.upper()

        for keyword in keywords:

            if keyword in upper_line:

                # Example:
                # NAME: RAHUL KUMAR
                parts = re.split(
                    r"[:\-]",
                    line,
                    maxsplit=1
                )

                if len(parts) == 2:

                    candidate = parts[1].strip()

                    if len(candidate.split()) >= 2:
                        return candidate

                # Otherwise check next line
                if i + 1 < len(lines):

                    candidate = lines[i + 1].strip()

                    if len(candidate.split()) >= 2:
                        return candidate

    return ""


# ---------------------------------------------------------
# ADDRESS EXTRACTION
# ---------------------------------------------------------

def extract_address(lines):
    address_keywords = [
        "ADDRESS",
        "RESIDENT",
        "VILLAGE",
        "STREET",
        "ROAD",
        "DISTRICT",
        "STATE",
        "PIN"
    ]

    address_lines = []

    collecting = False

    for line in lines:

        upper_line = line.upper()

        if "ADDRESS" in upper_line:
            collecting = True

            parts = re.split(
                r"[:\-]",
                line,
                maxsplit=1
            )

            if len(parts) == 2:
                address_lines.append(
                    parts[1].strip()
                )

            continue

        if collecting:

            if any(
                keyword in upper_line
                for keyword in address_keywords
            ):
                address_lines.append(line)

            elif address_lines:
                # Stop after the address section
                if len(address_lines) >= 4:
                    break

    return ", ".join(
        clean_lines(address_lines)
    )


# ---------------------------------------------------------
# OCR RESULT PARSER
# ---------------------------------------------------------

def parse_ocr_result(result):
    """
    Handle different RapidOCR result formats.
    """

    texts = []
    scores = []
    boxes = []

    if result is None:
        return texts, scores, boxes

    # Newer RapidOCR result object
    if hasattr(result, "txts"):

        texts = list(
            getattr(result, "txts", []) or []
        )

        scores = list(
            getattr(result, "scores", []) or []
        )

        boxes = list(
            getattr(result, "boxes", []) or []
        )

        return texts, scores, boxes

    # Tuple/list style result
    if isinstance(result, (tuple, list)):

        if len(result) >= 1:
            first = result[0]

            if isinstance(first, list):
                boxes = first

        if len(result) >= 2:

            second = result[1]

            if isinstance(second, list):

                for item in second:

                    if isinstance(item, (tuple, list)):

                        if len(item) >= 2:

                            texts.append(
                                str(item[0])
                            )

                            try:
                                scores.append(
                                    float(item[1])
                                )
                            except Exception:
                                pass

        return texts, scores, boxes

    return texts, scores, boxes


# ---------------------------------------------------------
# MAIN OCR FUNCTION
# ---------------------------------------------------------

def extract_text(image):
    """
    Main OCR function used by app.py.

    Returns a dictionary containing:
    document type
    name
    document number
    date of birth
    address
    confidence
    raw text
    """

    try:

        # Preprocess uploaded image
        processed_image = preprocess_image(image)

        # Get OCR engine
        engine = get_ocr_engine()

        # Run OCR
        result = engine(processed_image)

        # Parse result
        texts, scores, boxes = parse_ocr_result(result)

        # Clean OCR text
        lines = clean_lines(texts)

        raw_text = "\n".join(lines)

        # Calculate confidence
        valid_scores = []

        for score in scores:

            try:
                score = float(score)

                if 0 <= score <= 1:
                    valid_scores.append(score)

                elif 1 < score <= 100:
                    valid_scores.append(score / 100)

            except Exception:
                continue

        if valid_scores:
            confidence = round(
                sum(valid_scores) /
                len(valid_scores),
                3
            )
        else:
            confidence = 0.0

        # Extract fields
        document_type = detect_document_type(
            raw_text
        )

        document_number = extract_pan_number(
            raw_text
        )

        date_of_birth = extract_date(
            raw_text
        )

        name = extract_name(lines)

        address = extract_address(lines)

        # Return structured result
        return {
            "document_type": document_type,
            "name": name,
            "document_number": document_number,
            "date_of_birth": date_of_birth,
            "address": address,
            "confidence": confidence,
            "raw_text": raw_text
        }

    except Exception as e:

        return {
            "document_type": "Unknown",
            "name": "",
            "document_number": "",
            "date_of_birth": "",
            "address": "",
            "confidence": 0.0,
            "raw_text": "",
            "error": f"OCR Error: {str(e)}"
        }
