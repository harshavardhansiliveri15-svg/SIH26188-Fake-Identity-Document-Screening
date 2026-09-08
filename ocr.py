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

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_document_type(text):
    text = text.lower()

    pan_words = [
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

    if any(word in text for word in pan_words):
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
        match = re.search(pattern, text)

        if match:
            if match.lastindex:
                return match.group(1).upper()

            return match.group(0).upper()

    return "Not detected"


def valid_name(text):
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

    if not words:
        return False

    if len(words) > 6:
        return False

    letters = re.sub(
        r"[^A-Za-z]",
        "",
        text
    )

    if len(letters) < 3:
        return False

    return True


def clean_name(text):
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
    )

    return text.strip()


def is_name_label(text):
    text = clean_text(text).lower()

    text = re.sub(
        r"[^a-z ]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    labels = [
        "name",
        "full name",
        "given name",
        "given names",
        "first name",
        "surname",
        "last name",
        "family name"
    ]

    return text in labels


def is_dob_label(text):
    text = clean_text(text).lower()

    text = re.sub(
        r"[^a-z ]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    labels = [
        "dob",
        "date of birth",
        "birth date",
        "dateofbirth"
    ]

    return text in labels


def center_of_box(box):
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]

    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys)
    )


def box_height(box):
    ys = [point[1] for point in box]

    return max(ys) - min(ys)


def box_width(box):
    xs = [point[0] for point in box]

    return max(xs) - min(xs)


def extract_value_near_label(
    label_index,
    detections,
    max_vertical_gap=160,
    max_horizontal_gap=900
):
    label_box = detections[label_index][0]

    label_x, label_y = center_of_box(
        label_box
    )

    label_height = max(
        box_height(label_box),
        1
    )

    candidates = []

    for index, detection in enumerate(detections):
        if index == label_index:
            continue

        box, text, confidence = detection

        value = clean_text(text)

        if not value:
            continue

        x, y = center_of_box(box)

        vertical_distance = y - label_y

        horizontal_distance = abs(
            x - label_x
        )

        # Prefer text appearing below the label.
        if vertical_distance < -label_height * 0.8:
            continue

        if vertical_distance > max_vertical_gap:
            continue

        if horizontal_distance > max_horizontal_gap:
            continue

        score = (
            abs(vertical_distance) * 1.0
            + horizontal_distance * 0.25
            - float(confidence) * 80
        )

        candidates.append(
            (score, value, index)
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    if candidates:
        return candidates[0][1]

    return "Not detected"


def extract_name_from_layout(detections):
    # Method 1:
    # Look for a detected NAME label and inspect
    # nearby text boxes.

    for index, detection in enumerate(detections):
        text = clean_text(
            detection[1]
        )

        if is_name_label(text):
            candidate = extract_value_near_label(
                index,
                detections,
                max_vertical_gap=180,
                max_horizontal_gap=1000
            )

            candidate = clean_name(
                candidate
            )

            if valid_name(candidate):
                return candidate

    # Method 2:
    # Sometimes OCR detects "NAME: Alice Sharma"
    # as a single text box.

    for detection in detections:
        text = clean_text(
            detection[1]
        )

        match = re.search(
            r"(?:full\s+name|name)\s*[:\-]\s*"
            r"([A-Za-z][A-Za-z .'-]{2,70})",
            text,
            re.IGNORECASE
        )

        if match:
            candidate = clean_name(
                match.group(1)
            )

            if valid_name(candidate):
                return candidate

    # Method 3:
    # Try a line where NAME and the value were
    # merged by OCR.

    for detection in detections:
        text = clean_text(
            detection[1]
        )

        if "name" in text.lower():
            candidate = re.sub(
                r".*?\bname\b",
                "",
                text,
                flags=re.IGNORECASE
            )

            candidate = re.sub(
                r"^[\s:\-]+",
                "",
                candidate
            )

            candidate = clean_name(
                candidate
            )

            if valid_name(candidate):
                return candidate

    return "Not detected"


def extract_dob_from_layout(detections):
    # First try label-based extraction.

    for index, detection in enumerate(detections):
        text = clean_text(
            detection[1]
        )

        if is_dob_label(text):
            candidate = extract_value_near_label(
                index,
                detections,
                max_vertical_gap=180,
                max_horizontal_gap=1000
            )

            match = re.search(
                r"\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}",
                candidate
            )

            if match:
                return match.group(0)

            match = re.search(
                r"\d{1,2}\s+[A-Za-z]+\s+\d{2,4}",
                candidate
            )

            if match:
                return match.group(0)

    # Second try combined OCR text.

    combined = "\n".join(
        clean_text(item[1])
        for item in detections
    )

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
            combined,
            re.IGNORECASE
        )

        if match:
            return match.group(1)

    # Final fallback: any date-like text.

    for pattern in [
        r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b"
    ]:
        match = re.search(
            pattern,
            combined,
            re.IGNORECASE
        )

        if match:
            return match.group(0)

    return "Not detected"


def extract_document_number(text):
    patterns = [
        r"(?:document\s*(?:no|number))\s*[:\-]?\s*"
        r"([A-Z0-9\-]{4,30})",

        r"(?:id\s*(?:no|number))\s*[:\-]?\s*"
        r"([A-Z0-9\-]{4,30})",

        r"(?:passport\s*(?:no|number))\s*[:\-]?\s*"
        r"([A-Z0-9\-]{4,30})",

        r"(?:license|licence)\s*(?:no|number)"
        r"\s*[:\-]?\s*"
        r"([A-Z0-9\-]{4,30})"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).upper()

    return extract_pan_number(text)


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
            value = clean_text(
                match.group(1)
            )

            if value:
                return value

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

            if text and confidence >= 0.20:
                detections.append(
                    (
                        box,
                        text,
                        confidence
                    )
                )

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

        name = extract_name_from_layout(
            detections
        )

        date_of_birth = extract_dob_from_layout(
            detections
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
            "pan_number": "Not detected",
            "confidence": 0.0,
            "raw_text": "",
            "error": str(e)
        }
