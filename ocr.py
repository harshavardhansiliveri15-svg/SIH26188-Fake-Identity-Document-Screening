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


def clean_text(text):
    text = str(text)

    replacements = {
        "|": "I",
        "—": "-",
        "–": "-",
        "’": "'",
        "`": ""
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def detect_document_type(text):
    text = text.lower()

    if any(word in text for word in [
        "permanent account number",
        "permanent account",
        "income tax",
        "income-tax",
        "pan card",
        "pan no",
        "pan number",
        "tax department",
        "tax identity"
    ]):
        return "PAN / Tax Identity Card"

    if any(word in text for word in [
        "passport",
        "passport no",
        "passport number",
        "nationality"
    ]):
        return "Passport"

    if any(word in text for word in [
        "driver license",
        "driver's license",
        "driving license",
        "driving licence"
    ]):
        return "Driver License"

    if any(word in text for word in [
        "identity card",
        "identity document",
        "national id",
        "id card"
    ]):
        return "Identity Document"

    if any(word in text for word in [
        "birth certificate",
        "certificate of birth"
    ]):
        return "Birth Certificate"

    return "Unknown Document"


def extract_pan_number(text):
    text = text.upper()

    patterns = [
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        r"(?:PAN|P\.?A\.?N\.?)\s*(?:NO|NUMBER|CARD)?\s*[:\-]?\s*([A-Z]{5}[0-9]{4}[A-Z])"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text
        )

        if match:
            if match.lastindex:
                return match.group(1).upper()

            return match.group(0).upper()

    return "Not detected"


def extract_dob(text):
    patterns = [
        r"(?:date\s*of\s*birth|dob|birth\s*date)"
        r"\s*[:\-]?\s*"
        r"(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",

        r"(?:date\s*of\s*birth|dob|birth\s*date)"
        r"\s*[:\-]?\s*"
        r"(\d{1,2}\s+[A-Za-z]+\s+\d{2,4})"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1)

    fallback = re.search(
        r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b",
        text
    )

    if fallback:
        return fallback.group(0)

    return "Not detected"


def looks_like_name(text):
    text = clean_text(text)

    text = re.sub(
        r"[^A-Za-z .'-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = text.split()

    if len(words) < 1 or len(words) > 6:
        return False

    letters = re.sub(
        r"[^A-Za-z]",
        "",
        text
    )

    if len(letters) < 3:
        return False

    return True


def extract_name_regex(text):
    patterns = [
        r"(?:full\s*name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,70})",
        r"(?:name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,70})",
        r"(?:given\s*names?)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{1,60})",
        r"(?:first\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{1,60})",
        r"(?:surname|last\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{1,60})"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            candidate = clean_text(
                match.group(1)
            )

            if looks_like_name(candidate):
                return candidate

    return "Not detected"


def extract_name_from_boxes(detections):
    name_labels = [
        "name",
        "full name",
        "given name",
        "given names",
        "first name",
        "surname",
        "last name",
        "family name"
    ]

    for i, detection in enumerate(detections):
        box = detection["box"]
        text = clean_text(
            detection["text"]
        ).lower()

        if text not in name_labels:
            continue

        label_x = sum(
            point[0] for point in box
        ) / 4

        label_y = sum(
            point[1] for point in box
        ) / 4

        candidates = []

        for j, other in enumerate(detections):
            if i == j:
                continue

            other_box = other["box"]
            other_text = clean_text(
                other["text"]
            )

            other_x = sum(
                point[0] for point in other_box
            ) / 4

            other_y = sum(
                point[1] for point in other_box
            ) / 4

            vertical_distance = other_y - label_y
            horizontal_distance = abs(
                other_x - label_x
            )

            if vertical_distance < -30:
                continue

            if vertical_distance > 250:
                continue

            if horizontal_distance > 1200:
                continue

            if not looks_like_name(other_text):
                continue

            score = (
                abs(vertical_distance)
                + horizontal_distance * 0.25
                - other["confidence"] * 100
            )

            candidates.append(
                (
                    score,
                    other_text
                )
            )

        if candidates:
            candidates.sort(
                key=lambda x: x[0]
            )

            return candidates[0][1]

    return "Not detected"


def extract_address(text):
    patterns = [
        r"(?:address|residential\s+address)"
        r"\s*[:\-]\s*(.+)",

        r"(?:permanent\s+address)"
        r"\s*[:\-]\s*(.+)"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return clean_text(
                match.group(1)
            )

    return "Not detected"


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

        detections = []

        detected_text = []
        confidences = []

        for result in results:
            if len(result) < 3:
                continue

            box = result[0]

            text = clean_text(
                result[1]
            )

            confidence = float(
                result[2]
            )

            if not text:
                continue

            if confidence < 0.20:
                continue

            detections.append({
                "box": box,
                "text": text,
                "confidence": confidence
            })

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

        name = extract_name_from_boxes(
            detections
        )

        if name == "Not detected":
            name = extract_name_regex(
                normalized_text
            )

        date_of_birth = extract_dob(
            normalized_text
        )

        document_number = pan_number

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

        diagnostic_lines = []

        for index, detection in enumerate(
            detections,
            start=1
        ):
            diagnostic_lines.append(
                "OCR {}: {} | confidence {:.2f}".format(
                    index,
                    detection["text"],
                    detection["confidence"]
                )
            )

        diagnostic_text = "\n".join(
            diagnostic_lines
        )

        return {
            "document_type": document_type,
            "name": name,
            "document_number": document_number,
            "date_of_birth": date_of_birth,
            "address": address,
            "pan_number": pan_number,
            "confidence": confidence,
            "raw_text": full_text,
            "diagnostic_text": diagnostic_text
        }

    except Exception as e:
        return {
            "document_type": "OCR Error",
            "name": "Not detected",
            "document_number": "Not detected",
            "date_of_birth": "Not detected",
            "address": "Not detected",
            "pan_number": "Not detected",
            "confidence": 0.0,
            "raw_text": "",
            "diagnostic_text": "",
            "error": str(e)
        }
