def calculate_risk(
    ocr_result,
    validation_result,
    tamper_result,
    face_result
):
    """
    Risk engine.

    Later, the actual risk-scoring logic will be placed here.
    """

    return {
        "score": 18,
        "level": "LOW",
        "reason": (
            "The demonstration document received a LOW risk "
            "classification because the placeholder analysis "
            "results are positive."
        )
    }