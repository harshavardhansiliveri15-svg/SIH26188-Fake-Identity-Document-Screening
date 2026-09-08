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

    if width < 1600:
        scale = 1600 / width
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


def detect_document_type(text):
    text = text.lower()

    if any(x in text for x in [
        "permanent account number",
        "income tax",
        "income-tax",
        "pan card",
        "tax department",
        "tax identity"
    ]):
        return "PAN / Tax Identity Card"

    if any(x in text for x in [
        "passport",
        "passport no",
        "passport number",
        "nationality"
    ]):
        return "Passport"

    if any(x in text for x in [
        "driver license",
        "driver's license",
        "driving license",
        "driving licence",
        "license no",
        "licence no"
    ]):
        return "Driver License"

    if any(x in text for x in [
        "identity card",
        "identity document",
        "national id",
        "id card"
    ]):
        return "Identity Document"

    if any(x in text for x in [
        "birth certificate",
        "certificate of birth"
    ]):
        return "Birth Certificate"

    if any(x in text for x in [
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
            value = re.sub(r"\s+", " ", value)

            if value:
                return value

    return "Not detected"


def extract_pan_number(text):
    patterns = [
        r"(?:pan\s*(?:no|number)?|permanent\s*account\s*number)\s*[:\-]?\s*([A-Z]{5}[0-9]{4}[A-Z])",
        r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).upper()

    return "Not detected"


def extract_name(text):
    patterns = [
        r"(?:full\s*name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,60})",
        r"(?:name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,60})",
        r"(?:given\s*name|given\s*names|first\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{1,50})",
        r"(?:surname|last\s*name|family\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{1,40})"
    ]

    return extract_field(
        text,
        patterns
    )


def extract_parent_name(text):
    patterns = [
        r"(?:father'?s?\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{2,60})",
        r"(?:father)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,60})",
        r"(?:parent'?s?\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{2,60})"
    ]

    return extract_field(
        text,
        patterns
    )


def extract_date_of_birth(text):
    patterns = [
        r"(?:date\s*of\s*birth|dob|birth\s*date)\s*[:\-]?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
        r"(?:date\s*of\s*birth|dob|birth\s*date)\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{2,4})",
        r"(?:date\s*of\s*birth|dob|birth\s*date)\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{2,4})"
    ]

    return extract_field(
        text,
        patterns
    )


def extract_document_number(text):
    patterns = [
        r"(?:document\s*(?:no|number))\s*[:\-]?\s*([A-Z0-9\-]{4,30})",
        r"(?:id\s*(?:no|number))\s*[:\-]?\s*([A-Z0-9\-]{4,30})",
        r"(?:passport\s*(?:no|number))\s*[:\-]?\s*([A-Z0-9\-]{4,30})",
        r"(?:license|licence)\s*(?:no|number)\s*[:\-]?\s*([A-Z0-9\-]{4,30})"
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
        r"(?:address|residential\s+address)\s*[:\-]\s*(.+)",
        r"(?:permanent\s+address)\s*[:\-]\s*(.+)"
    ]

    return extract_field(
        text,
        patterns
    )


def extract_text(document_image):
    try:
        processed_image = preprocess_image(
            document_image
        )

        reader = get_reader()

        results = reader.readtext(
            processed_image,
            detail=1,
            paragraph=False
        )

        detected_text = []
        confidences = []

        for result in results:
            if len(result) < 3:
                continue

            text = str(
                result[1]
            ).strip()

            confidence = float(
                result[2]
            )

            if text and confidence >= 0.20:
                detected_text.append(text)
                confidences.append(confidence)

        full_text = "\n".join(
            detected_text
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

        normalized_text = re.sub(
            r"[ \t]+",
            " ",
            full_text
        )

        document_type = detect_document_type(
            normalized_text
        )

        name = extract_name(
            normalized_text
        )

        parent_name = extract_parent_name(
            normalized_text
        )

        date_of_birth = extract_date_of_birth(
            normalized_text
        )

        document_number = extract_document_number(
            normalized_text
        )

        address = extract_address(
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
