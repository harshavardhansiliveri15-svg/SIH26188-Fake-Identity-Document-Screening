import re
import cv2
import numpy as np
from PIL import Image
from rapidocr import RapidOCR


_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine

    if _ocr_engine is None:
        _ocr_engine = RapidOCR()

    return _ocr_engine


def convert_to_bgr(image):
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
                cv2.COLOR_BGRA2BGR
            )

        return image

    image = cv2.imread(str(image))

    if image is None:
        raise ValueError(
            "Could not read document image."
        )

    return image


def preprocess_image(image):
    image = convert_to_bgr(image)

    if image is None:
        raise ValueError(
            "Could not read document image."
        )

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


def normalize_text(text):
    text = str(text or "")

    text = text.replace("|", " ")
    text = text.replace("—", "-")
    text = text.replace("–", "-")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


def detect_document_type(text):
    text_lower = text.lower()

    pan_words = [
        "permanent account number",
        "permanent account",
        "income tax",
        "income-tax",
        "income tax department",
        "pan card",
        "pan no",
        "pan number"
    ]

    if any(
        word in text_lower
        for word in pan_words
    ):
        return "PAN / Tax Identity Card"

    if re.search(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        text.upper()
    ):
        return "PAN / Tax Identity Card"

    if any(
        word in text_lower
        for word in [
            "passport",
            "passport no",
            "passport number",
            "nationality"
        ]
    ):
        return "Passport"

    if any(
        word in text_lower
        for word in [
            "driver license",
            "driver's license",
            "driving license",
            "driving licence",
            "license no",
            "licence no"
        ]
    ):
        return "Driver License"

    if any(
        word in text_lower
        for word in [
            "identity card",
            "identity document",
            "national id",
            "id card"
        ]
    ):
        return "Identity Document"

    if any(
        word in text_lower
        for word in [
            "birth certificate",
            "certificate of birth"
        ]
    ):
        return "Birth Certificate"

    return "Unknown Document"


def clean_ocr_value(value):
    value = str(value or "").strip()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    value = value.strip(
        " :-|,.;"
    )

    return value


def looks_like_person_name(value):
    value = clean_ocr_value(value)

    if len(value) < 5:
        return False

    if len(value) > 70:
        return False

    if re.search(
        r"\d",
        value
    ):
        return False

    words = value.split()

    if not 2 <= len(words) <= 6:
        return False

    blocked = {
        "income",
        "income tax",
        "department",
        "india",
        "permanent",
        "account",
        "number",
        "signature",
        "date",
        "birth",
        "father",
        "father name",
        "name",
        "pan",
        "government",
        "govt",
        "of",
        "tax"
    }

    lowered = value.lower()

    if lowered in blocked:
        return False

    for word in words:

        if not re.fullmatch(
            r"[A-Za-z][A-Za-z.'-]*",
            word
        ):
            return False

        # Reject extremely short OCR fragments
        if len(word) < 2:
            return False

    return True


def extract_pan_number(text):
    text_upper = text.upper()

    patterns = [
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",

        r"(?:PAN|P\.?A\.?N\.?)"
        r"\s*(?:NO|NUMBER|CARD)?"
        r"\s*[:\-]?\s*"
        r"([A-Z]{5}[0-9]{4}[A-Z])"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text_upper
        )

        for match in matches:

            if isinstance(match, tuple):
                match = match[0]

            candidate = re.sub(
                r"[^A-Z0-9]",
                "",
                str(match)
            )

            if re.fullmatch(
                r"[A-Z]{5}[0-9]{4}[A-Z]",
                candidate
            ):
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

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return clean_ocr_value(
                match.group(1)
            )

    return "Not detected"


def extract_name_from_lines(lines):
    name_labels = [
        "name",
        "full name",
        "full-name",
        "surname",
        "given name",
        "given names",
        "first name"
    ]

    father_labels = [
        "father",
        "father name",
        "father's name",
        "father s name"
    ]

    # --------------------------------------------------
    # STEP 1:
    # Look specifically for a NAME label.
    # --------------------------------------------------

    for index, line in enumerate(lines):

        clean_line = clean_ocr_value(line)

        lower_line = clean_line.lower()

        for label in name_labels:

            if lower_line.startswith(label):

                remainder = re.sub(
                    rf"^{re.escape(label)}"
                    r"\s*[:\-]?\s*",
                    "",
                    clean_line,
                    flags=re.IGNORECASE
                )

                if looks_like_person_name(
                    remainder
                ):
                    return remainder

                # Sometimes OCR places the name
                # on the line immediately below "Name"

                if index + 1 < len(lines):

                    next_line = clean_ocr_value(
                        lines[index + 1]
                    )

                    if looks_like_person_name(
                        next_line
                    ):
                        return next_line

    # --------------------------------------------------
    # STEP 2:
    # Conservative fallback.
    #
    # Only accept a name containing at least
    # two proper-looking words.
    # --------------------------------------------------

    for line in lines:

        clean_line = clean_ocr_value(line)

        if not clean_line:
            continue

        lower_line = clean_line.lower()

        # Ignore father-related lines
        if any(
            label in lower_line
            for label in father_labels
        ):
            continue

        if not looks_like_person_name(
            clean_line
        ):
            continue

        # Reject one-word OCR fragments
        if len(clean_line.split()) < 2:
            continue

        upper_line = clean_line.upper()

        blocked_lines = [
            "PERMANENT ACCOUNT NUMBER",
            "INCOME TAX DEPARTMENT",
            "GOVT OF INDIA",
            "GOVERNMENT OF INDIA",
            "INCOME TAX",
            "PERMANENT ACCOUNT"
        ]

        if upper_line in blocked_lines:
            continue

        return clean_line

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

            value = clean_ocr_value(
                match.group(1)
            )

            if len(value) >= 5:
                return value

    return "Not detected"


def extract_text(document_image):

    try:

        processed_image = preprocess_image(
            document_image
        )

        engine = get_ocr_engine()

        result = engine(
            processed_image
        )

        texts = []
        scores = []

        if result is not None:

            if hasattr(result, "txts"):

                texts = list(
                    result.txts or []
                )

            if hasattr(result, "scores"):

                scores = list(
                    result.scores or []
                )

            if isinstance(result, tuple):

                if len(result) >= 2:

                    first = result[0]
                    second = result[1]

                    if isinstance(first, list):
                        texts = first

                    if isinstance(second, list):
                        scores = second

        detected_text = []

        for text in texts:

            text = clean_ocr_value(
                text
            )

            if text:
                detected_text.append(
                    text
                )

        full_text = "\n".join(
            detected_text
        )

        normalized_text = normalize_text(
            full_text
        )

        # --------------------------------------------------
        # OCR CONFIDENCE
        # --------------------------------------------------

        confidence = 0.0

        if scores:

            valid_scores = []

            for score in scores:

                try:

                    value = float(score)

                    if 0 <= value <= 1:

                        valid_scores.append(
                            value
                        )

                except Exception:
                    pass

            if valid_scores:

                confidence = round(
                    (
                        sum(valid_scores)
                        / len(valid_scores)
                    ) * 100,
                    2
                )

        # --------------------------------------------------
        # DOCUMENT TYPE
        # --------------------------------------------------

        document_type = detect_document_type(
            normalized_text
        )

        # --------------------------------------------------
        # PAN NUMBER
        # --------------------------------------------------

        pan_number = extract_pan_number(
            normalized_text
        )

        if pan_number != "Not detected":

            document_type = (
                "PAN / Tax Identity Card"
            )

        # --------------------------------------------------
        # DATE OF BIRTH
        # --------------------------------------------------

        date_of_birth = extract_date_of_birth(
            normalized_text
        )

        # --------------------------------------------------
        # NAME
        # --------------------------------------------------

        lines = [
            clean_ocr_value(x)
            for x in detected_text
            if clean_ocr_value(x)
        ]

        name = extract_name_from_lines(
            lines
        )

        # --------------------------------------------------
        # ADDRESS
        # --------------------------------------------------

        address = extract_address(
            normalized_text
        )

        # --------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------

        return {
            "document_type": document_type,
            "name": name,
            "document_number": pan_number,
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
            "confidence": 0.0,
            "raw_text": "",
            "error": str(e)
        }
