import re
import cv2
import numpy as np
import easyocr
from PIL import Image

_reader = None


def get_reader():
    global _reader

    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)

    return _reader


def convert_to_bgr(image):
    if isinstance(image, Image.Image):
        image = np.array(image)

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if isinstance(image, np.ndarray):
        image = image.copy()

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return image

    return cv2.imread(str(image))


def preprocess_image(image):
    image = convert_to_bgr(image)

    if image is None:
        raise ValueError("Could not read document image.")

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

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    return gray


def clean_ocr_text(text):
    text = str(text)

    replacements = {
        "|": "I",
        "—": "-",
        "–": "-",
        "’": "'",
        "`": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def detect_document_type(text):
    text_lower = text.lower()

    pan_keywords = [
        "permanent account number",
        "permanent account",
        "income tax",
        "income-tax",
        "pan card",
        "pan no",
        "pan number",
        "tax department",
        "tax identity"
    ]

    if any(keyword in text_lower for keyword in pan_keywords):
        return "PAN / Tax Identity Card"

    if any(keyword in text_lower for keyword in [
        "passport",
        "passport no",
        "passport number",
        "nationality"
    ]):
        return "Passport"

    if any(keyword in text_lower for keyword in [
        "driver license",
        "driver's license",
        "driving license",
        "driving licence",
        "license no",
        "licence no"
    ]):
        return "Driver License"

    if any(keyword in text_lower for keyword in [
        "identity card",
        "identity document",
        "national id",
        "id card"
    ]):
        return "Identity Document"

    if any(keyword in text_lower for keyword in [
        "birth certificate",
        "certificate of birth"
    ]):
        return "Birth Certificate"

    if any(keyword in text_lower for keyword in [
        "form w-4",
        "w-4",
        "withholding certificate"
    ]):
        return "Tax Form / W-4"

    return "Unknown Document"


def extract_field(text, patterns):
    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            value = match.group(1).strip()

            value = re.sub(
                r"\s+",
                " ",
                value
            )

            if value:
                return value

    return "Not detected"


def extract_pan_number(text):
    normalized = text.upper()

    patterns = [
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        r"(?:PAN|P\.?A\.?N\.?)\s*(?:NO|NUMBER|CARD)?\s*[:\-]?\s*([A-Z]{5}[0-9]{4}[A-Z])",
        r"(?:PERMANENT\s+ACCOUNT\s+NUMBER)\s*[:\-]?\s*([A-Z]{5}[0-9]{4}[A-Z])"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            normalized,
            re.IGNORECASE
        )

        if match:
            if match.lastindex:
                return match.group(1).upper()

            return match.group(0).upper()

    return "Not detected"


def normalize_name(value):
    value = clean_ocr_text(value)

    value = re.sub(
        r"^(name|full name)\s*[:\-]?\s*",
        "",
        value,
        flags=re.IGNORECASE
    )

    value = re.sub(
        r"[^A-Za-z .'-]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    value = value.strip()

    words = value.split()

    if not words:
        return "Not detected"

    if len(words) > 6:
        words = words[:6]

    return " ".join(words)


def looks_like_name(value):
    if not value or value == "Not detected":
        return False

    value = normalize_name(value)

    words = value.split()

    if len(words) < 1 or len(words) > 6:
        return False

    if any(char.isdigit() for char in value):
        return False

    letters = re.sub(
        r"[^A-Za-z]",
        "",
        value
    )

    return len(letters) >= 3


def extract_name(text):
    patterns = [
        r"(?:full\s*name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,70})",
        r"(?:name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,70})",
        r"(?:name)\s+([A-Za-z][A-Za-z .'-]{2,70})",
        r"(?:given\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{1,60})",
        r"(?:given\s*names?)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{1,60})",
        r"(?:first\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{1,60})",
        r"(?:surname)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{1,60})",
        r"(?:last\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{1,60})",
        r"(?:family\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{1,60})"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            candidate = normalize_name(
                match.group(1)
            )

            if looks_like_name(candidate):
                return candidate

    return "Not detected"


def extract_parent_name(text):
    patterns = [
        r"(?:father'?s?\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{2,70})",
        r"(?:father)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,70})",
        r"(?:parent'?s?\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{2,70})"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            candidate = normalize_name(
                match.group(1)
            )

            if looks_like_name(candidate):
                return candidate

    return "Not detected"


def extract_date_of_birth(text):
    patterns = [
        r"(?:date\s*of\s*birth|dob|birth\s*date)"
        r"\s*[:\-]?\s*"
        r"(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",

        r"(?:date\s*of\s*birth|dob|birth\s*date)"
        r"\s*[:\-]?\s*"
        r"(\d{1,2}\s+[A-Za-z]+\s+\d{2,4})",

        r"(?:date\s*of\s*birth|dob|birth\s*date)"
        r"\s*[:\-]?\s*"
        r"([A-Za-z]+\s+\d{1,2},?\s+\d{2,4})"
    ]

    return extract_field(
        text,
        patterns
    )


def extract_date_fallback(text):
    patterns = [
        r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}\b"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(0).strip()

    return "Not detected"


def extract_document_number(text):
    patterns = [
        r"(?:document\s*(?:no|number))"
        r"\s*[:\-]?\s*"
        r"([A-Z0-9\-]{4,30})",

        r"(?:id\s*(?:no|number))"
        r"\s*[:\-]?\s*"
        r"([A-Z0-9\-]{4,30})",

        r"(?:passport\s*(?:no|number))"
        r"\s*[:\-]?\s*"
        r"([A-Z0-9\-]{4,30})",

        r"(?:license|licence)\s*(?:no|number)"
        r"\s*[:\-]?\s*"
        r"([A-Z0-9\-]{4,30})"
    ]

    result = extract_field(
        text,
        patterns
    )

    if result != "Not detected":
        return result

    return extract_pan_number(text)


def extract_address(text):
    patterns = [
        r"(?:address|residential\s+address)"
        r"\s*[:\-]\s*(.+)",

        r"(?:permanent\s+address)"
        r"\s*[:\-]\s*(.+)"
    ]

    return extract_field(
        text,
        patterns
    )


def build_combined_text(results):
    lines = []

    for result in results:
        if len(result) < 3:
            continue

        text = clean_ocr_text(
            result[1]
        )

        if text:
            lines.append(text)

    return lines


def extract_text(document_image):
    try:
        processed_image = preprocess_image(
            document_image
        )

        reader = get_reader()

        results = reader.readtext(
            processed_image,
            detail=1,
            paragraph=False,
            contrast_ths=0.05,
            adjust_contrast=0.7,
            text_threshold=0.5,
            low_text=0.2,
            link_threshold=0.3
        )

        detected_text = []
        confidences = []

        for result in results:
            if len(result) < 3:
                continue

            text = clean_ocr_text(
                result[1]
            )

            confidence = float(
                result[2]
            )

            if text and confidence >= 0.20:
                detected_text.append(text)
                confidences.append(confidence)

        full_text = "\n".join(
            detected_text
        )

        normalized_text = re.sub(
            r"[ \t]+",
            " ",
            full_text
        )

        document_type = detect_document_type(
            normalized_text
        )

        pan_number = extract_pan_number(
            normalized_text
        )

        if (
            pan_number != "Not detected"
            and document_type == "Unknown Document"
        ):
            document_type = "PAN / Tax Identity Card"

        name = extract_name(
            normalized_text
        )

        parent_name = extract_parent_name(
            normalized_text
        )

        date_of_birth = extract_date_of_birth(
            normalized_text
        )

        if date_of_birth == "Not detected":
            date_of_birth = extract_date_fallback(
                normalized_text
            )

        document_number = extract_document_number(
            normalized_text
        )

        address = extract_address(
            normalized_text
        )

        if confidences:
            confidence = round(
                (
                    sum(confidences)
                    / len(confidences)
                ) * 100,
                2
            )
        else:
            confidence = 0.0

        return {
            "document_type": document_type,
            "name": name,
            "document_number": document_number,
            "date_of_birth": date_of_birth,
            "address": address,
            "parent_name": parent_name,
            "pan_number": pan_number,
            "confidence": confidence,
            "raw_text": full_text
        }

    except Exception as e:
        return {
            "document_type": "OCR Error",
            "name": "Not detected",
            "document_number": "Not detected",
            "date_of_birth": "Not detected",
            "address": "Not detected",
            "parent_name": "Not detected",
            "pan_number": "Not detected",
            "confidence": 0.0,
            "raw_text": "",
            "error": str(e)
        }
