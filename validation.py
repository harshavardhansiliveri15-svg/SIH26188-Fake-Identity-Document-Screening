import re


def validate_document(ocr_result):
    """
    Validate the document using OCR-extracted information.

    This is a screening layer for a prototype.
    It does not provide official document authentication.
    """

    if not ocr_result:
        return {
            "status": "REVIEW",
            "message": "No OCR result available."
        }

    document_type = ocr_result.get("document_type", "Unknown Document")
    name = ocr_result.get("name", "Not detected")
    document_number = ocr_result.get(
        "document_number",
        "Not detected"
    )
    date_of_birth = ocr_result.get(
        "date_of_birth",
        "Not detected"
    )
    confidence = float(
        ocr_result.get("confidence", 0)
    )

    checks = []
    score = 0

    # ---------------------------------
    # 1. Document type
    # ---------------------------------

    if document_type != "Unknown Document":
        checks.append("Document type detected")
        score += 1
    else:
        checks.append("Document type could not be determined")

    # ---------------------------------
    # 2. Name
    # ---------------------------------

    if name != "Not detected" and len(name) >= 3:
        checks.append("Name field detected")
        score += 1
    else:
        checks.append("Name field not detected")

    # ---------------------------------
    # 3. Document number
    # ---------------------------------

    if document_number != "Not detected":

        cleaned_number = re.sub(
            r"[^A-Za-z0-9]",
            "",
            str(document_number)
        )

        if len(cleaned_number) >= 4:
            checks.append("Document number detected")
            score += 1
        else:
            checks.append("Document number appears incomplete")

    else:
        checks.append("Document number not detected")

    # ---------------------------------
    # 4. Date
    # ---------------------------------

    if date_of_birth != "Not detected":

        if re.search(
            r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}",
            str(date_of_birth)
        ):
            checks.append("Date format appears valid")
            score += 1
        else:
            checks.append("Date format could not be confirmed")

    else:
        checks.append("Date field not detected")

    # ---------------------------------
    # 5. OCR confidence
    # ---------------------------------

    if confidence >= 70:
        checks.append("OCR confidence is good")
        score += 1

    elif confidence >= 40:
        checks.append("OCR confidence is moderate")

    else:
        checks.append("OCR confidence is low")

    # ---------------------------------
    # Overall result
    # ---------------------------------

    if score >= 4:
        status = "PASS"
        message = "Document passed basic screening checks."

    elif score >= 2:
        status = "REVIEW"
        message = "Document requires additional review."

    else:
        status = "REVIEW"
        message = "Insufficient information for reliable validation."

    return {
        "status": status,
        "message": message,
        "checks": checks,
        "score": score,
        "max_score": 5
    }
