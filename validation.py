import re


def validate_document(ocr_result, document_image=None):
    """
    Basic document validation/screening.

    The second argument is kept for compatibility with the
    existing Streamlit application.

    This is an AI-assisted screening prototype and does not
    provide official document authentication.
    """

    try:
        if not ocr_result:
            return {
                "status": "REVIEW",
                "message": "No OCR result available.",
                "checks": [],
                "score": 0,
                "max_score": 5
            }

        document_type = ocr_result.get(
            "document_type",
            "Unknown Document"
        )

        name = ocr_result.get(
            "name",
            "Not detected"
        )

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

        # -----------------------------
        # DOCUMENT TYPE
        # -----------------------------

        if document_type not in [
            "Unknown Document",
            "OCR Error"
        ]:
            checks.append("✓ Document type detected")
            score += 1
        else:
            checks.append("⚠ Document type not determined")

        # -----------------------------
        # NAME
        # -----------------------------

        if (
            name != "Not detected"
            and len(str(name).strip()) >= 3
        ):
            checks.append("✓ Name field detected")
            score += 1
        else:
            checks.append("⚠ Name field not detected")

        # -----------------------------
        # DOCUMENT NUMBER
        # -----------------------------

        if document_number != "Not detected":

            cleaned_number = re.sub(
                r"[^A-Za-z0-9]",
                "",
                str(document_number)
            )

            if len(cleaned_number) >= 4:
                checks.append(
                    "✓ Document number detected"
                )
                score += 1
            else:
                checks.append(
                    "⚠ Document number appears incomplete"
                )

        else:
            checks.append(
                "⚠ Document number not detected"
            )

        # -----------------------------
        # DATE
        # -----------------------------

        if date_of_birth != "Not detected":

            date_text = str(date_of_birth)

            if re.search(
                r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}",
                date_text
            ):
                checks.append(
                    "✓ Date format detected"
                )
                score += 1
            else:
                checks.append(
                    "⚠ Date format could not be confirmed"
                )

        else:
            checks.append(
                "⚠ Date field not detected"
            )

        # -----------------------------
        # OCR CONFIDENCE
        # -----------------------------

        if confidence >= 70:
            checks.append(
                "✓ OCR confidence is good"
            )
            score += 1

        elif confidence >= 40:
            checks.append(
                "⚠ OCR confidence is moderate"
            )

        else:
            checks.append(
                "⚠ OCR confidence is low"
            )

        # -----------------------------
        # FINAL VALIDATION
        # -----------------------------

        if score >= 4:
            status = "PASS"
            message = (
                "Document passed basic screening checks."
            )

        elif score >= 2:
            status = "REVIEW"
            message = (
                "Document requires additional review."
            )

        else:
            status = "REVIEW"
            message = (
                "Insufficient information for reliable validation."
            )

        return {
            "status": status,
            "message": message,
            "checks": checks,
            "score": score,
            "max_score": 5
        }

    except Exception as e:

        return {
            "status": "REVIEW",
            "message": "Validation could not be completed.",
            "checks": [
                "Validation module encountered an error."
            ],
            "score": 0,
            "max_score": 5,
            "error": str(e)
        }
