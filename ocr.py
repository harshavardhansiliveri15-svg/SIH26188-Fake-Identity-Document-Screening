```python
"""
OCR EXTRACTION MODULE V2
For SIH26188 - AI-Powered Fake Identity & Document Screening

Supports:
- PAN / Tax Identity style mock documents
- Driver License style documents
- Passport style documents
- Identity Card style documents
- Birth Certificate style documents
- General OCR extraction

This module performs preliminary OCR extraction only.
It does not prove document authenticity.
"""

import re
import cv2
import numpy as np
import easyocr
from PIL import Image


# ============================================================
# EASY OCR READER
# ============================================================

_reader = None


def get_reader():
    global _reader

    if _reader is None:
        _reader = easyocr.Reader(
            ["en"],
            gpu=False
        )

    return _reader


# ============================================================
# IMAGE CONVERSION
# ============================================================

def _convert_to_bgr(image):
    """
    Convert PIL or NumPy image to OpenCV BGR format.
    """

    if isinstance(image, Image.Image):

        image = np.array(image)

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

    if isinstance(image, np.ndarray):

        image = image.copy()

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

    image = cv2.imread(str(image))

    return image


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Prepare image for OCR.
    """

    image = _convert_to_bgr(image)

    if image is None:
        raise ValueError(
            "Could not read document image."
        )

    height, width = image.shape[:2]

    # Upscale smaller images
    if width < 1600:

        scale = 1600 / width

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    # Mild denoising
    image = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    return gray


# ============================================================
# DOCUMENT TYPE DETECTION
# ============================================================

def detect_document_type(text):

    text_lower = text.lower()

    # PAN / Tax identity
    pan_keywords = [
        "income tax",
        "income-tax",
        "permanent account number",
        "pan card",
        "pan",
        "tax department",
        "tax identity",
        "govt of india",
        "government of india"
    ]

    if any(
        keyword in text_lower
        for keyword in pan_keywords
    ):
        return "PAN / Tax Identity Card"

    # Passport
    passport_keywords = [
        "passport",
        "passport no",
        "passport number",
        "nationality"
    ]

    if any(
        keyword in text_lower
        for keyword in passport_keywords
    ):
        return "Passport"

    # Driver license
    license_keywords = [
        "driver license",
        "driver's license",
        "driving license",
        "driving licence",
        "dl no",
        "licence no"
    ]

    if any(
        keyword in text_lower
        for keyword in license_keywords
    ):
        return "Driver License"

    # Identity card
    identity_keywords = [
        "identity card",
        "identity document",
        "national id",
        "id card"
    ]

    if any(
        keyword in text_lower
        for keyword in identity_keywords
    ):
        return "Identity Document"

    # Birth certificate
    birth_keywords = [
        "birth certificate",
        "certificate of birth",
        "date of birth certificate"
    ]

    if any(
        keyword in text_lower
        for keyword in birth_keywords
    ):
        return "Birth Certificate"

    # Tax form
    tax_keywords = [
        "form w-4",
        "w-4",
        "withholding certificate",
        "employee's withholding"
    ]

    if any(
        keyword in text_lower
        for keyword in tax_keywords
    ):
        return "Tax Form / W-4"

    return "Unknown Document"


# ============================================================
# GENERIC FIELD EXTRACTION
# ============================================================

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


# ============================================================
# PAN-LIKE IDENTIFIER
# ============================================================

def extract_pan_number(text):
    """
    Detect a PAN-like identifier.

    This is only a pattern detector for mock/demo documents.
    """

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

            value = match.group(1)

            return value.upper()

    return "Not detected"


# ============================================================
# NAME EXTRACTION
# ============================================================

def extract_name(text):

    patterns = [

        # Full Name: ALICE SHARMA
        r"(?:full\s*name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,60})",

        # Name: ALICE SHARMA
        r"(?:name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,60})",

        # Name - ALICE SHARMA
        r"(?:name)\s*[-]\s*([A-Za-z][A-Za-z .'-]{2,60})",

        # Given name
        r"(?:given\s*names?|given\s*name|first\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{1,50})",

        # Surname / family name
        r"(?:surname|last\s*name|family\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{1,40})"
    ]

    name = extract_field(
        text,
        patterns
    )

    return name


# ============================================================
# FATHER / PARENT NAME
# ============================================================

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


# ============================================================
# DATE OF BIRTH
# ============================================================

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


# ============================================================
# DOCUMENT NUMBER
# ============================================================

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

    # Try PAN-style identifier
    return extract_pan_number(text)


# ============================================================
# ADDRESS
# ============================================================

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


# ============================================================
# OCR EXTRACTION
# ============================================================

def extract_text(document_image):

    try:

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        processed_image = preprocess_image(
            document_image
        )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        reader = get_reader()

        results = reader.readtext(
            processed_image,
            detail=1,
            paragraph=False
        )

        detected_text = []
        confidences = []

        # ----------------------------------------------------
        # COLLECT OCR RESULTS
        # ----------------------------------------------------

        for result in results:

            if len(result) < 3:
                continue

            text = str(
                result[1]
            ).strip()

            confidence = float(
                result[2]
            )

            if not text:
                continue

            # Keep moderately confident OCR text
            if confidence >= 0.20:

                detected_text.append(
                    text
                )

                confidences.append(
                    confidence
                )

        # ----------------------------------------------------
        # BUILD FULL TEXT
        # ----------------------------------------------------

        full_text = "\n".join(
            detected_text
        )

        # ----------------------------------------------------
        # OCR CONFIDENCE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # NORMALIZED TEXT
        # ----------------------------------------------------

        normalized_text = re.sub(
            r"[ \t]+",
            " ",
            full_text
        )

        # ----------------------------------------------------
        # DOCUMENT TYPE
        # ----------------------------------------------------

        document_type = (
            detect_document_type(
                normalized_text
            )
        )

        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        name = extract_name(
            normalized_text
        )

        # ----------------------------------------------------
        # PARENT NAME
        # ----------------------------------------------------

        parent_name = extract_parent_name(
            normalized_text
        )

        # ----------------------------------------------------
        # DOB
        # ----------------------------------------------------

        date_of_birth = (
            extract_date_of_birth(
                normalized_text
            )
        )

        # ----------------------------------------------------
        # DOCUMENT NUMBER
        # ----------------------------------------------------

        document_number = (
            extract_document_number(
                normalized_text
            )
        )

        # ----------------------------------------------------
        # ADDRESS
        # ----------------------------------------------------

        address = extract_address(
            normalized_text
        )

        # ----------------------------------------------------
        # PAN FALLBACK
        # ----------------------------------------------------

        pan_number = extract_pan_number(
            normalized_text
        )

        # If a PAN-like number is detected,
        # classify as PAN/Tax Identity.
        if (
            pan_number != "Not detected"
            and document_type == "Unknown Document"
        ):

            document_type = (
                "PAN / Tax Identity Card"
            )

        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

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


# ============================================================
# OPTIONAL STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    test_image = "test_image.png"

    print("=" * 60)
    print("OCR EXTRACTION MODULE V2")
    print("=" * 60)

    try:

        result = extract_text(
            test_image
        )

        print(
            "Document Type:",
            result.get(
                "document_type"
            )
        )

        print(
            "Name:",
            result.get(
                "name"
            )
        )

        print(
            "Document Number:",
            result.get(
                "document_number"
            )
        )

        print(
            "Date of Birth:",
            result.get(
                "date_of_birth"
            )
        )

        print(
            "Parent Name:",
            result.get(
                "parent_name"
            )
        )

        print(
            "PAN Number:",
            result.get(
                "pan_number"
            )
        )

        print(
            "OCR Confidence:",
            result.get(
                "confidence"
            ),
            "%"
        )

        print()
        print("Raw OCR Text:")
        print("-" * 60)
        print(
            result.get(
                "raw_text",
                ""
            )
        )

        print("=" * 60)

    except Exception as e:

        print(
            "OCR Test Error:",
            e
        )
```
