import re
import cv2
import numpy as np
import easyocr


# Load OCR reader once instead of creating it for every upload
_reader = None


def get_reader():
    global _reader

    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)

    return _reader


def preprocess_image(image):
    """
    Prepare uploaded image for OCR.
    """

    # Convert PIL image / array into OpenCV format
    image = np.array(image)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize small images
    height, width = image.shape[:2]

    if width < 1200:
        scale = 1200 / width
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    return gray


def detect_document_type(text):
    """
    Estimate document type from OCR text.
    """

    text_lower = text.lower()

    if any(word in text_lower for word in [
        "driver license",
        "driver's license",
        "driving licence",
        "driving license"
    ]):
        return "Driver License"

    if any(word in text_lower for word in [
        "passport",
        "passport no",
        "nationality"
    ]):
        return "Passport"

    if any(word in text_lower for word in [
        "identity card",
        "identity document",
        "national id",
        "id card"
    ]):
        return "Identity Document"

    if any(word in text_lower for word in [
        "employee's withholding",
        "withholding certificate",
        "form w-4",
        "w-4"
    ]):
        return "Tax Form / W-4"

    if any(word in text_lower for word in [
        "birth certificate",
        "certificate of birth"
    ]):
        return "Birth Certificate"

    return "Unknown Document"


def extract_field(text, patterns):
    """
    Try multiple regex patterns and return the first match.
    """

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            value = match.group(1).strip()

            if value:
                return value

    return "Not detected"


def extract_text(document_image):
    """
    Perform OCR on the uploaded document image.

    Returns the same dictionary structure expected by app.py.
    """

    try:
        processed_image = preprocess_image(document_image)

        reader = get_reader()

        results = reader.readtext(
            processed_image,
            detail=1
        )

        detected_text = []
        confidences = []

        for result in results:

            if len(result) >= 3:

                text = str(result[1]).strip()
                confidence = float(result[2])

                if text and confidence >= 0.30:
                    detected_text.append(text)
                    confidences.append(confidence)

        full_text = "\n".join(detected_text)

        # Average OCR confidence
        if confidences:
            confidence = round(
                (sum(confidences) / len(confidences)) * 100,
                2
            )
        else:
            confidence = 0

        document_type = detect_document_type(full_text)

        # Name extraction
        name = extract_field(
            full_text,
            [
                r"(?:full\s*name|name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,60})",
                r"(?:employee'?s?\s+name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,60})"
            ]
        )

        # Document number extraction
        document_number = extract_field(
            full_text,
            [
                r"(?:document\s*(?:no|number)|id\s*(?:no|number)|passport\s*(?:no|number))\s*[:\-]?\s*([A-Z0-9\-]{4,30})",
                r"(?:license\s*(?:no|number))\s*[:\-]?\s*([A-Z0-9\-]{4,30})"
            ]
        )

        # Date of birth extraction
        date_of_birth = extract_field(
            full_text,
            [
                r"(?:date\s*of\s*birth|dob|birth\s*date)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
                r"(?:date\s*of\s*birth|dob|birth\s*date)\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{2,4})"
            ]
        )

        # Address extraction
        address = extract_field(
            full_text,
            [
                r"(?:address|residential\s+address)\s*[:\-]\s*(.+)",
                r"(?:permanent\s+address)\s*[:\-]\s*(.+)"
            ]
        )

        return {
            "document_type": document_type,
            "name": name,
            "document_number": document_number,
            "date_of_birth": date_of_birth,
            "address": address,
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
            "confidence": 0,
            "raw_text": "",
            "error": str(e)
        }
