def validate_document(ocr_result, document_image=None):
    """
    Basic AI-assisted document screening.
    Compatible with the existing app.py interface.
    """

    try:
        # Support both dictionary and text OCR results
        if isinstance(ocr_result, dict):

            document_type = str(
                ocr_result.get("document_type", "Unknown")
            )

            name = str(
                ocr_result.get("name", "Not detected")
            )

            document_number = str(
                ocr_result.get("document_number", "Not detected")
            )

            date_of_birth = str(
                ocr_result.get("date_of_birth", "Not detected")
            )

            confidence = float(
                ocr_result.get("confidence", 0)
            )

        else:
            text = str(ocr_result or "").strip()

            document_type = (
                "Document detected"
                if text
                else "Unknown Document"
            )

            name = "Not detected"
            document_number = "Not detected"
            date_of_birth = "Not detected"

            confidence = 80 if text else 0

        # --------------------------------
        # Screening checks
        # --------------------------------

        checks = []
        passed = 0
        total = 0

        # Document presence
        total += 1

        if document_type not in [
            "",
            "Unknown",
            "Unknown Document",
            "OCR Error"
        ]:
            checks.append("✓ Document content detected")
            passed += 1
        else:
            checks.append("⚠ Document type not determined")

        # OCR confidence
        total += 1

        if confidence >= 70:
            checks.append("✓ OCR confidence is good")
            passed += 1

        elif confidence >= 40:
            checks.append("⚠ OCR confidence is moderate")

        else:
            checks.append("⚠ OCR confidence is low")

        # Name
        total += 1

        if (
            name
            and name.lower() not in [
                "not detected",
                "none",
                "unknown"
            ]
            and len(name.strip()) >= 3
        ):
            checks.append("✓ Name information detected")
            passed += 1
        else:
            checks.append("⚠ Name information not detected")

        # Document number
        total += 1

        if (
            document_number
            and document_number.lower() not in [
                "not detected",
                "none",
                "unknown"
            ]
            and len(document_number.strip()) >= 4
        ):
            checks.append("✓ Document number detected")
            passed += 1
        else:
            checks.append("⚠ Document number not detected")

        # Date
        total += 1

        if (
            date_of_birth
            and date_of_birth.lower() not in [
                "not detected",
                "none",
                "unknown"
            ]
        ):
            checks.append("✓ Date information detected")
            passed += 1
        else:
            checks.append("⚠ Date information not detected")

        # --------------------------------
        # Final screening result
        # --------------------------------

        percentage = round(
            (passed / total) * 100
        )

        if percentage >= 70:
            status = "PASS"
            message = (
                "Document passed the basic screening checks."
            )

        elif percentage >= 40:
            status = "REVIEW"
            message = (
                "Document contains usable information "
                "but requires additional review."
            )

        else:
            status = "REVIEW"
            message = (
                "Insufficient information for reliable "
                "validation."
            )

        return {
            "status": status,
            "message": message,
            "checks": checks,
            "score": passed,
            "max_score": total,
            "percentage": percentage
        }

    except Exception as e:

        return {
            "status": "REVIEW",
            "message": "Validation module encountered an error.",
            "checks": [],
            "score": 0,
            "max_score": 5,
            "percentage": 0,
            "error": str(e)
        }
