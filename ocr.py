import re
import cv2
import numpy as np
from PIL import Image
from rapidocr import RapidOCR


# =========================================================
# OCR ENGINE
# =========================================================

_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine

    if _ocr_engine is None:
        _ocr_engine = RapidOCR()

    return _ocr_engine


# =========================================================
# IMAGE CONVERSION
# =========================================================

def convert_to_bgr(image):

    if isinstance(image, Image.Image):
        image = np.array(image)

    if not isinstance(image, np.ndarray):
        raise ValueError("Unsupported image format")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    raise ValueError(f"Unsupported image shape: {image.shape}")


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    image = convert_to_bgr(image)

    height, width = image.shape[:2]

    if width < 1600:

        scale = 1600 / width

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        image,
        -1,
        kernel
    )

    return sharpened


# =========================================================
# TEXT CLEANING
# =========================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_lines(lines):

    cleaned = []

    for line in lines:

        line = normalize_text(line)

        if line:
            cleaned.append(line)

    return cleaned


# =========================================================
# DOCUMENT TYPE
# =========================================================

def detect_document_type(text):

    t = text.upper()

    if (
        "INCOME TAX DEPARTMENT" in t
        or "PERMANENT ACCOUNT NUMBER" in t
        or re.search(
            r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
            t
        )
    ):
        return "PAN Card"

    if (
        "AADHAAR" in t
        or "UNIQUE IDENTIFICATION AUTHORITY" in t
        or "UNIQUE IDENTIFICATION" in t
    ):
        return "Aadhaar Card"

    if "ELECTION COMMISSION" in t:
        return "Voter ID"

    if (
        "DRIVING LICENCE" in t
        or "DRIVING LICENSE" in t
    ):
        return "Driving Licence"

    if "PASSPORT" in t:
        return "Passport"

    return "Unknown"


# =========================================================
# DOCUMENT NUMBER
# =========================================================

def extract_document_number(text, document_type):

    t = text.upper()

    # PAN
    pan = re.search(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        t
    )

    if pan:
        return pan.group(0)

    # Aadhaar
    aadhaar = re.search(
        r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        t
    )

    if aadhaar:

        return re.sub(
            r"\s+",
            " ",
            aadhaar.group(0)
        )

    return ""


# =========================================================
# DATE VALIDATION
# =========================================================

def is_valid_date(date_string):

    if not date_string:
        return False

    date_string = date_string.strip()

    patterns = [
        r"^\d{2}[/-]\d{2}[/-]\d{4}$",
        r"^\d{4}[/-]\d{2}[/-]\d{2}$",
        r"^\d{2}[/-]\d{2}[/-]\d{2}$"
    ]

    return any(
        re.match(pattern, date_string)
        for pattern in patterns
    )


# =========================================================
# DOB EXTRACTION
# =========================================================

def extract_date(text, lines=None):

    # -----------------------------------------------------
    # First: search specifically for DOB labels
    # -----------------------------------------------------

    dob_patterns = [

        # DOB: 18/05/1990
        r"(?:DATE\s*OF\s*BIRTH|DOB|D\.O\.B\.?)"
        r"\s*[:\-]?\s*"
        r"(\d{2}[/-]\d{2}[/-]\d{4})",

        # DATE OF BIRTH 18-05-1990
        r"(?:DATE\s*OF\s*BIRTH|DOB|D\.O\.B\.?)"
        r"\s+"
        r"(\d{2}[/-]\d{2}[/-]\d{4})",

        # DOB: 1990/05/18
        r"(?:DATE\s*OF\s*BIRTH|DOB|D\.O\.B\.?)"
        r"\s*[:\-]?\s*"
        r"(\d{4}[/-]\d{2}[/-]\d{2})"
    ]

    upper_text = text.upper()

    for pattern in dob_patterns:

        match = re.search(
            pattern,
            upper_text
        )

        if match:

            candidate = match.group(1)

            if is_valid_date(candidate):
                return candidate


    # -----------------------------------------------------
    # Second: check individual OCR lines
    # -----------------------------------------------------

    if lines:

        for i, line in enumerate(lines):

            upper_line = line.upper()

            if (
                "DATE OF BIRTH" in upper_line
                or "DOB" in upper_line
                or "D.O.B" in upper_line
            ):

                # Date on same line
                match = re.search(
                    r"\d{2}[/-]\d{2}[/-]\d{4}",
                    line
                )

                if match:
                    return match.group(0)

                # Date on next OCR line
                if i + 1 < len(lines):

                    match = re.search(
                        r"\d{2}[/-]\d{2}[/-]\d{4}",
                        lines[i + 1]
                    )

                    if match:
                        return match.group(0)


    # -----------------------------------------------------
    # Final fallback
    #
    # Only used if no DOB label was detected.
    # -----------------------------------------------------

    date_patterns = [
        r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
        r"\b\d{4}[/-]\d{2}[/-]\d{2}\b"
    ]

    for pattern in date_patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            candidate = match.group(0)

            if is_valid_date(candidate):
                return candidate

    return ""


# =========================================================
# NAME
# =========================================================

def extract_name(lines):

    keywords = [
        "NAME",
        "FULL NAME",
        "GIVEN NAME",
        "SURNAME"
    ]

    for i, line in enumerate(lines):

        upper = line.upper().strip()

        for keyword in keywords:

            if upper.startswith(keyword):

                # NAME: ABC
                parts = re.split(
                    r"[:\-]",
                    line,
                    maxsplit=1
                )

                if len(parts) == 2:

                    candidate = normalize_text(
                        parts[1]
                    )

                    if candidate:
                        return candidate

                # NAME
                # ABC
                if i + 1 < len(lines):

                    candidate = normalize_text(
                        lines[i + 1]
                    )

                    if candidate:
                        return candidate

    return ""


# =========================================================
# ADDRESS
# =========================================================

def extract_address(lines):

    address_lines = []

    collecting = False

    stop_keywords = [
        "DATE OF BIRTH",
        "DOB",
        "GENDER",
        "SEX",
        "SIGNATURE",
        "PAN",
        "PERMANENT ACCOUNT NUMBER"
    ]

    for line in lines:

        upper = line.upper().strip()

        if "ADDRESS" in upper:

            collecting = True

            parts = re.split(
                r"[:\-]",
                line,
                maxsplit=1
            )

            if len(parts) == 2:

                value = normalize_text(
                    parts[1]
                )

                if value:
                    address_lines.append(value)

            continue

        if collecting:

            if any(
                keyword in upper
                for keyword in stop_keywords
            ):
                break

            if line.strip():

                address_lines.append(
                    normalize_text(line)
                )

            if len(address_lines) >= 5:
                break

    return ", ".join(
        clean_lines(address_lines)
    )


# =========================================================
# RAPIDOCR RESULT
# =========================================================

def parse_ocr_result(result):

    if result is None:
        return [], [], []

    texts = getattr(
        result,
        "txts",
        None
    )

    scores = getattr(
        result,
        "scores",
        None
    )

    boxes = getattr(
        result,
        "boxes",
        None
    )

    if texts is None:
        texts = []

    if scores is None:
        scores = []

    if boxes is None:
        boxes = []

    return (
        list(texts),
        list(scores),
        list(boxes)
    )


# =========================================================
# MAIN OCR FUNCTION
# =========================================================

def extract_text(image):

    try:

        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------

        processed_image = preprocess_image(
            image
        )

        # -------------------------------------------------
        # OCR
        # -------------------------------------------------

        engine = get_ocr_engine()

        result = engine(
            processed_image
        )

        # -------------------------------------------------
        # PARSE
        # -------------------------------------------------

        texts, scores, boxes = parse_ocr_result(
            result
        )

        # -------------------------------------------------
        # CLEAN
        # -------------------------------------------------

        lines = clean_lines(
            texts
        )

        raw_text = "\n".join(
            lines
        )

        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        valid_scores = []

        for score in scores:

            try:

                score = float(score)

                if 0 <= score <= 1:

                    valid_scores.append(
                        score
                    )

                elif 1 < score <= 100:

                    valid_scores.append(
                        score / 100
                    )

            except Exception:
                pass

        if valid_scores:

            confidence = round(
                sum(valid_scores)
                / len(valid_scores),
                3
            )

        else:

            confidence = 0.0

        # -------------------------------------------------
        # DOCUMENT TYPE
        # -------------------------------------------------

        document_type = detect_document_type(
            raw_text
        )

        # -------------------------------------------------
        # FIELDS
        # -------------------------------------------------

        document_number = extract_document_number(
            raw_text,
            document_type
        )

        date_of_birth = extract_date(
            raw_text,
            lines
        )

        name = extract_name(
            lines
        )

        address = extract_address(
            lines
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

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
