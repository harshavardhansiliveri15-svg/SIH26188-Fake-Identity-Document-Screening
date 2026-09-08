import re


def validate_document(ocr_result, document_image=None):
    """
    AI-assisted document screening.

    Compatible with OCR dictionaries and OCR text strings.
    The image argument is retained for compatibility with app.py.
    """

    try:
        # -----------------------------------------
        # Handle OCR result supplied as a dictionary
        # -----------------------------------------

        if isinstance(ocr_result, dict):

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

            raw_text = ocr_result.get(
                "raw_text",
                ""
            )

        # -----------------------------------------
        # Handle OCR result supplied as plain text
        # -----------------------------------------

        else:

            raw_text = str(ocr_result or "")

            text_lower = raw_text.lower()

            if "passport" in text_lower:
                document_type = "Passport"

            elif (
                "driver license" in text_lower
                or "driving licence" in text_lower
                or "driving license" in text_lower
            ):
                document_type = "Driver License"

            elif (
                "identity card" in text_lower
                or "identity document" in text_lower
                or "national id" in text_lower
                or "id card" in text_lower
            ):
                document_type = "Identity Document"

            elif (
                "w-4" in text_lower
                or "withholding certificate" in text_lower
            ):
                document_type = "Tax Form / W-4"

            else:
                document_type = "Unknown Document"

            # Try to find a name
            name_match = re.search(
                r"(?:name|full name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{2,60})",
                raw_text,
                re.IGNORECASE
            )

            name = (
                name_match.group(1).strip()
                if name_match
                else "Not detected"
            )

            # Try to find a document number
            number_match = re.search(
                r"(?:document\s*(?:no|number)|id\s*(?:no|number)|passport\s*(?:no|number)|license\s*(?:no|number))\s*[:\-]?\s*([A-Za-z0-9\-]{4,30})",
                raw_text,
                re.IGNORECASE
            )

            document_number = (
                number_match.group(1).strip()
                if number_match
                else "Not detected"
            )

            # Try to find a date
            date_match = re.search(
                r"(?:date of birth|dob|birth date)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
                raw_text,
                re.IGNORECASE
            )

            date_of_birth = (
                date_match.group(1).strip()
                if date_match
                else "Not detected"
            )

            # Estimate confidence from whether OCR produced text
            confidence = 80 if raw_text.strip() else 0

        # -----------------------------------------
        # Validation checks
        # -----------------------------------------

        checks = []
        score = 0

        # Document type
        if document_type not in [
            "Unknown Document",
            "OCR Error"
        ]:
            checks.append("✓ Document type detected")
            score += 1
        else:
            checks.append("⚠ Document type not determined")

        # Name
        if (
            name != "Not detected"
            and len(str(name).strip()) >= 3
        ):
            checks.append("✓ Name field detected")
            score += 1
        else:
            checks.append("⚠ Name field not detected")

        # Document number
        if document_number != "Not detected":

            cleaned = re.sub(
                r"[^A-Za-z0-9]",
                "",
                str(document_number)
            )

            if len(cleaned) >= 4:
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

        # Date
        if date_of_birth != "Not detected":

            if re.search(
                r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}",
                str(date_of_birth)
            ):
                checks.append(
                    "✓ Date format detected"
                )
                score += 1
            else:
                checks.append(
                    "⚠ Date format uncertain"
                )
        else:
            checks.append(
                "⚠ Date field not detected"
            )

        # OCR confidence
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

        # -----------------------------------------
        # Final status
        # -----------------------------------------

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
            "checks": [],
            "score": 0,
            "max_score": 5,
            "error": str(e)
        }
