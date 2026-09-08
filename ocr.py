```python
import re
import cv2
import numpy as np
from PIL import Image
from rapidocr import RapidOCR


# ============================================================
# OCR ENGINE
# ============================================================

_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine

    if _ocr_engine is None:
        _ocr_engine = RapidOCR()

    return _ocr_engine


# ============================================================
# IMAGE CONVERSION
# ============================================================

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


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = convert_to_bgr(image)

    if image is None:
        raise ValueError(
            "Could not read document image."
        )

    height, width = image.shape[:2]

    # Enlarge small images
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

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    return gray


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_text(text):

    text = str(text or "")

    text = text.replace("|", " ")
    text = text.replace("—", "-")
    text = text.replace("–", "-")

    # Normalize spaces around date separators
    text = re.sub(
        r"\s*([\/\-.])\s*",
        r"\1",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


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


# ============================================================
# DOCUMENT TYPE
# ============================================================

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


# ============================================================
# PAN NUMBER
# ============================================================

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


# ============================================================
# DATE OF BIRTH
# ============================================================

def normalize_date_candidate(value):

    """
    Cleans common OCR errors inside a date.
    """

    value = clean_ocr_value(value)

    # Remove spaces around separators
    value = re.sub(
        r"\s*([\/\-.])\s*",
        r"\1",
        value
    )

    # Common OCR character mistakes in numeric dates
    value = value.replace("O", "0")
    value = value.replace("o", "0")

    value = value.replace("I", "1")
    value = value.replace("l", "1")

    return value


def is_valid_numeric_date(value):

    """
    Basic validation for DD/MM/YYYY-style dates.
    """

    value = normalize_date_candidate(value)

    match = re.fullmatch(
        r"(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})",
        value
    )

    if not match:
        return False

    try:

        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))

        if year < 100:
            year += 2000

        if not (
            1 <= day <= 31
            and
            1 <= month <= 12
            and
            1900 <= year <= 2100
        ):
            return False

        return True

    except Exception:
        return False


def extract_date_of_birth(text):

    if not text:
        return "Not detected"

    text = str(text)

    # --------------------------------------------------------
    # 1. Normalize common OCR mistakes
    # --------------------------------------------------------

    corrected = text

    corrections = {

        "D0B": "DOB",
        "D8B": "DOB",
        "D.O.B": "DOB",
        "D.O.B.": "DOB",
        "D O B": "DOB",

        "DATE 0F BIRTH": "DATE OF BIRTH",
        "DATE 0F B1RTH": "DATE OF BIRTH",
        "DATE OF B1RTH": "DATE OF BIRTH",
        "DATE OF BIRTH": "DATE OF BIRTH",

        "B1RTH": "BIRTH",
        "BIRTH": "BIRTH"
    }

    for old, new in corrections.items():

        corrected = re.sub(
            re.escape(old),
            new,
            corrected,
            flags=re.IGNORECASE
        )

    # Normalize spaces around date separators
    corrected = re.sub(
        r"\s*([\/\-.])\s*",
        r"\1",
        corrected
    )

    # --------------------------------------------------------
    # 2. Date formats
    # --------------------------------------------------------

    numeric_date = (
        r"\d{1,2}"
        r"\s*[\/\-.]\s*"
        r"\d{1,2}"
        r"\s*[\/\-.]\s*"
        r"\d{2,4}"
    )

    text_date = (
        r"(?:"
        r"\d{1,2}\s+[A-Za-z]{3,12}\s+\d{2,4}"
        r"|"
        r"[A-Za-z]{3,12}\s+\d{1,2},?\s+\d{2,4}"
        r")"
    )

    date_pattern = (
        rf"({numeric_date}|{text_date})"
    )

    # --------------------------------------------------------
    # 3. Explicit DOB + date
    # --------------------------------------------------------

    dob_patterns = [

        rf"(?:DATE\s*OF\s*BIRTH)"
        rf"\s*[:\-]?\s*"
        rf"{date_pattern}",

        rf"(?:DOB)"
        rf"\s*[:\-]?\s*"
        rf"{date_pattern}",

        rf"(?:BIRTH\s*DATE)"
        rf"\s*[:\-]?\s*"
        rf"{date_pattern}",

        rf"(?:D\s*[O0]\s*B)"
        rf"\s*[:\-]?\s*"
        rf"{date_pattern}",
    ]

    for pattern in dob_patterns:

        match = re.search(
            pattern,
            corrected,
            re.IGNORECASE
        )

        if match:

            value = normalize_date_candidate(
                match.group(1)
            )

            if (
                is_valid_numeric_date(value)
                or
                re.search(
                    r"[A-Za-z]",
                    value
                )
            ):
                return value

    # --------------------------------------------------------
    # 4. Split OCR lines
    # --------------------------------------------------------

    lines = [
        clean_ocr_value(line)
        for line in corrected.splitlines()
        if clean_ocr_value(line)
    ]

    for index, line in enumerate(lines):

        lower_line = line.lower()

        dob_label_found = any(
            label in lower_line
            for label in [
                "date of birth",
                "dob",
                "birth date",
                "date 0f birth",
                "date of b1rth",
                "birth"
            ]
        )

        if not dob_label_found:
            continue

        # ----------------------------------------------------
        # Same line
        # ----------------------------------------------------

        match = re.search(
            date_pattern,
            line,
            re.IGNORECASE
        )

        if match:

            value = normalize_date_candidate(
                match.group(1)
            )

            if (
                is_valid_numeric_date(value)
                or
                re.search(
                    r"[A-Za-z]",
                    value
                )
            ):
                return value

        # ----------------------------------------------------
        # Next line
        # ----------------------------------------------------

        if index + 1 < len(lines):

            next_line = lines[
                index + 1
            ]

            match = re.search(
                date_pattern,
                next_line,
                re.IGNORECASE
            )

            if match:

                value = normalize_date_candidate(
                    match.group(1)
                )

                if (
                    is_valid_numeric_date(value)
                    or
                    re.search(
                        r"[A-Za-z]",
                        value
                    )
                ):
                    return value

        # ----------------------------------------------------
        # Second following line
        # ----------------------------------------------------

        if index + 2 < len(lines):

            next_line = lines[
                index + 2
            ]

            match = re.search(
                date_pattern,
                next_line,
                re.IGNORECASE
            )

            if match:

                value = normalize_date_candidate(
                    match.group(1)
                )

                if (
                    is_valid_numeric_date(value)
                    or
                    re.search(
                        r"[A-Za-z]",
                        value
                    )
                ):
                    return value

    # --------------------------------------------------------
    # 5. Fuzzy DOB detection
    # --------------------------------------------------------

    fuzzy_patterns = [

        rf"D\s*[O0]\s*B"
        rf"\s*[:\-]?\s*"
        rf"{date_pattern}",

        rf"DATE\s*[O0]\s*F\s*BIRTH"
        rf"\s*[:\-]?\s*"
        rf"{date_pattern}",

        rf"DATE\s*OF\s*B[1I]RTH"
        rf"\s*[:\-]?\s*"
        rf"{date_pattern}",

        rf"D[O0][B8]"
        rf"\s*[:\-]?\s*"
        rf"{date_pattern}",
    ]

    for pattern in fuzzy_patterns:

        match = re.search(
            pattern,
            corrected,
            re.IGNORECASE
        )

        if match:

            value = normalize_date_candidate(
                match.group(1)
            )

            if (
                is_valid_numeric_date(value)
                or
                re.search(
                    r"[A-Za-z]",
                    value
                )
            ):
                return value

    # --------------------------------------------------------
    # 6. Find all possible dates
    # --------------------------------------------------------

    all_matches = list(
        re.finditer(
            date_pattern,
            corrected,
            re.IGNORECASE
        )
    )

    # --------------------------------------------------------
    # 7. Prefer date close to DOB/BIRTH wording
    # --------------------------------------------------------

    for match in all_matches:

        date_value = normalize_date_candidate(
            match.group(1)
        )

        start = max(
            0,
            match.start() - 150
        )

        end = min(
            len(corrected),
            match.end() + 150
        )

        nearby_text = corrected[
            start:end
        ].lower()

        if any(
            keyword in nearby_text
            for keyword in [
                "dob",
                "date of birth",
                "birth date",
                "birth"
            ]
        ):

            if (
                is_valid_numeric_date(date_value)
                or
                re.search(
                    r"[A-Za-z]",
                    date_value
                )
            ):
                return date_value

    # --------------------------------------------------------
    # 8. Controlled fallback
    #
    # Only accept a plausible full numeric date.
    # This prevents random numbers such as PAN digits
    # from being treated as DOB.
    # --------------------------------------------------------

    for match in all_matches:

        date_value = normalize_date_candidate(
            match.group(1)
        )

        if is_valid_numeric_date(
            date_value
        ):
            return date_value

    return "Not detected"


# ============================================================
# NAME VALIDATION
# ============================================================

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
        "permanent account",
        "account",
        "number",
        "signature",
        "date",
        "date of birth",
        "birth",
        "father",
        "father name",
        "father's name",
        "father s name",
        "name",
        "pan",
        "government",
        "govt",
        "govt of india",
        "government of india",
        "of",
        "tax",
        "card",
        "identity",
        "identity card",
        "tax department"
    }

    lowered = value.lower()

    if lowered in blocked:
        return False

    blocked_phrases = [

        "income tax department",
        "government of india",
        "govt of india",
        "permanent account number",
        "income tax",
        "permanent account"
    ]

    for phrase in blocked_phrases:

        if phrase in lowered:
            return False

    for word in words:

        if not re.fullmatch(
            r"[A-Za-z][A-Za-z.'-]*",
            word
        ):
            return False

        if len(word) < 2:
            return False

    return True


# ============================================================
# NAME LABEL DETECTION
# ============================================================

def is_name_label(text):

    text = clean_ocr_value(
        text
    ).lower()

    text = text.replace(
        "nane",
        "name"
    )

    text = text.replace(
        "neme",
        "name"
    )

    text = text.replace(
        "narne",
        "name"
    )

    text = text.replace(
        "na me",
        "name"
    )

    text = text.strip(
        " :-."
    )

    labels = [

        "name",
        "full name",
        "full-name",
        "given name",
        "given names",
        "first name",
        "surname"
    ]

    return text in labels


# ============================================================
# NAME EXTRACTION
# ============================================================

def extract_name_from_lines(lines):

    cleaned_lines = [

        clean_ocr_value(line)
        for line in lines
        if clean_ocr_value(line)
    ]

    # --------------------------------------------------------
    # METHOD 1: Explicit NAME label
    # --------------------------------------------------------

    for index, line in enumerate(
        cleaned_lines
    ):

        if is_name_label(line):

            remainder = re.sub(

                r"^(?:name|full\s*name|given\s*name|"
                r"given\s*names|first\s*name|surname)"
                r"\s*[:\-]?\s*",

                "",

                line,

                flags=re.IGNORECASE
            )

            if looks_like_person_name(
                remainder
            ):
                return remainder

            if index + 1 < len(
                cleaned_lines
            ):

                next_line = cleaned_lines[
                    index + 1
                ]

                if looks_like_person_name(
                    next_line
                ):
                    return next_line

    # --------------------------------------------------------
    # METHOD 2: OCR-damaged NAME label
    # --------------------------------------------------------

    for index, line in enumerate(
        cleaned_lines
    ):

        lower = line.lower()

        possible_name_label = any(

            x in lower

            for x in [

                "name:",
                "name -",
                "name ",
                "nane",
                "neme",
                "narne"
            ]
        )

        if possible_name_label:

            remainder = re.sub(

                r".*?(?:name|nane|neme|narne)"
                r"\s*[:\-]?\s*",

                "",

                line,

                flags=re.IGNORECASE
            )

            remainder = clean_ocr_value(
                remainder
            )

            if looks_like_person_name(
                remainder
            ):
                return remainder

            if index + 1 < len(
                cleaned_lines
            ):

                next_line = cleaned_lines[
                    index + 1
                ]

                if looks_like_person_name(
                    next_line
                ):
                    return next_line

    # --------------------------------------------------------
    # METHOD 3: Conservative fallback
    # --------------------------------------------------------

    candidates = []

    for index, line in enumerate(
        cleaned_lines
    ):

        if not looks_like_person_name(
            line
        ):
            continue

        lower = line.lower()

        if any(
            word in lower
            for word in [
                "father",
                "father's",
                "father s"
            ]
        ):
            continue

        if any(
            phrase in lower
            for phrase in [
                "income tax",
                "government of india",
                "govt of india",
                "permanent account",
                "department",
                "signature"
            ]
        ):
            continue

        score = 0

        word_count = len(
            line.split()
        )

        if 2 <= word_count <= 4:
            score += 3

        if any(
            c.isupper()
            for c in line
        ):
            score += 1

        nearby = " ".join(

            cleaned_lines[
                max(0, index - 2):
                min(
                    len(cleaned_lines),
                    index + 3
                )
            ]
        ).lower()

        if "date" in nearby:
            score += 1

        if "birth" in nearby:
            score += 1

        if "father" in nearby:
            score += 1

        if "pan" in nearby:
            score += 1

        candidates.append(
            (score, index, line)
        )

    if candidates:

        candidates.sort(

            key=lambda x: (
                x[0],
                -x[1]
            ),

            reverse=True
        )

        best_score = candidates[0][0]
        best_name = candidates[0][2]

        if best_score >= 4:
            return best_name

    return "Not detected"


# ============================================================
# ADDRESS EXTRACTION
# ============================================================

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


# ============================================================
# RAPIDOCR RESULT PARSER
# ============================================================

def parse_ocr_result(result):

    texts = []
    scores = []
    boxes = []

    if result is None:
        return texts, scores, boxes

    if hasattr(
        result,
        "txts"
    ):

        try:

            texts = list(
                result.txts or []
            )

        except Exception:

            texts = []

    if hasattr(
        result,
        "scores"
    ):

        try:

            scores = list(
                result.scores or []
            )

        except Exception:

            scores = []

    if hasattr(
        result,
        "boxes"
    ):

        try:

            boxes = list(
                result.boxes or []
            )

        except Exception:

            boxes = []

    if isinstance(
        result,
        tuple
    ):

        if len(result) >= 1:

            first = result[0]

            if isinstance(
                first,
                list
            ):

                boxes = first

        if len(result) >= 2:

            second = result[1]

            if isinstance(
                second,
                list
            ):

                texts = second

        if len(result) >= 3:

            third = result[2]

            if isinstance(
                third,
                list
            ):

                scores = third

    return texts, scores, boxes


# ============================================================
# MAIN OCR FUNCTION
# ============================================================

def extract_text(document_image):

    try:

        processed_image = preprocess_image(
            document_image
        )

        engine = get_ocr_engine()

        result = engine(
            processed_image
        )

        texts, scores, boxes = parse_ocr_result(
            result
        )

        detected_text = []

        valid_scores = []

        # ----------------------------------------------------
        # CLEAN OCR OUTPUT
        # ----------------------------------------------------

        for text in texts:

            text = clean_ocr_value(
                text
            )

            if text:

                detected_text.append(
                    text
                )

        # ----------------------------------------------------
        # OCR CONFIDENCE
        # ----------------------------------------------------

        for score in scores:

            try:

                value = float(score)

                if 0 <= value <= 1:

                    valid_scores.append(
                        value
                    )

                elif 1 < value <= 100:

                    valid_scores.append(
                        value / 100
                    )

            except Exception:

                pass

        confidence = 0.0

        if valid_scores:

            confidence = round(

                (
                    sum(valid_scores)
                    /
                    len(valid_scores)
                ) * 100,

                2
            )

        # ----------------------------------------------------
        # FULL TEXT
        # ----------------------------------------------------

        full_text = "\n".join(
            detected_text
        )

        normalized_text = normalize_text(
            full_text
        )

        # ----------------------------------------------------
        # DOCUMENT TYPE
        # ----------------------------------------------------

        document_type = detect_document_type(
            normalized_text
        )

        # ----------------------------------------------------
        # PAN NUMBER
        # ----------------------------------------------------

        pan_number = extract_pan_number(
            normalized_text
        )

        if pan_number != "Not detected":

            document_type = (
                "PAN / Tax Identity Card"
            )

        # ----------------------------------------------------
        # DATE OF BIRTH
        # ----------------------------------------------------

        date_of_birth = extract_date_of_birth(
            full_text
        )

        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        lines = [

            clean_ocr_value(x)

            for x in detected_text

            if clean_ocr_value(x)
        ]

        name = extract_name_from_lines(
            lines
        )

        # ----------------------------------------------------
        # ADDRESS
        # ----------------------------------------------------

        address = extract_address(
            normalized_text
        )

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        return {

            "document_type":
                document_type,

            "name":
                name,

            "document_number":
                pan_number,

            "date_of_birth":
                date_of_birth,

            "address":
                address,

            "confidence":
                confidence,

            "raw_text":
                full_text
        }

    except Exception as e:

        return {

            "document_type":
                "OCR Error",

            "name":
                "Not detected",

            "document_number":
                "Not detected",

            "date_of_birth":
                "Not detected",

            "address":
                "Not detected",

            "confidence":
                0.0,

            "raw_text":
                "",

            "error":
                str(e)
        }
```
