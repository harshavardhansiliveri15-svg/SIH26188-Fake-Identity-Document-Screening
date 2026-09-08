import streamlit as st
from PIL import Image

# ============================================================
# PROJECT MODULES
# ============================================================

from ocr import extract_text
from validation import validate_document
from tamper import detect_tampering
from face import verify_face
from risk_engine import calculate_risk


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Identity Screening | SIH26188",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL SIH CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f4f7fb;
}

/* Remove excessive top space */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main Header */
.hero {
    background: linear-gradient(135deg, #101b3d 0%, #2448a5 100%);
    padding: 34px 38px;
    border-radius: 20px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 25px rgba(30, 60, 120, 0.15);
}

.hero-title {
    font-size: 36px;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 12px;
}

.hero-subtitle {
    font-size: 16px;
    opacity: 0.95;
    margin-bottom: 8px;
}

.hero-description {
    font-size: 15px;
    line-height: 1.6;
    opacity: 0.9;
}


/* Section heading */
.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #16213e;
    margin-top: 28px;
    margin-bottom: 16px;
}


/* Upload cards */
.upload-card {
    background: white;
    border: 1px solid #dfe5ef;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 14px rgba(20, 40, 80, 0.05);
}


/* Result cards */
.result-card {
    background: white;
    border: 1px solid #dfe5ef;
    border-radius: 16px;
    padding: 22px;
    margin-top: 16px;
    box-shadow: 0 4px 14px rgba(20, 40, 80, 0.05);
}


/* Risk card */
.risk-card {
    background: white;
    border: 1px solid #dfe5ef;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(20, 40, 80, 0.08);
}


/* Risk score */
.risk-score {
    font-size: 54px;
    font-weight: 800;
    color: #16213e;
    margin: 5px 0;
}

.risk-label {
    font-size: 15px;
    color: #667085;
}


/* Status cards */
.status-card {
    background: white;
    border: 1px solid #dfe5ef;
    border-radius: 15px;
    padding: 18px;
    min-height: 125px;
    box-shadow: 0 4px 12px rgba(20, 40, 80, 0.05);
}

.status-title {
    font-size: 14px;
    color: #667085;
    margin-bottom: 8px;
}

.status-value {
    font-size: 22px;
    font-weight: 750;
    color: #16213e;
}

.status-detail {
    font-size: 13px;
    color: #667085;
    margin-top: 6px;
}


/* Pipeline */
.pipeline-card {
    background: white;
    border: 1px solid #dfe5ef;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 4px 14px rgba(20, 40, 80, 0.05);
}


/* Demo warning */
.demo-box {
    background: #fff7df;
    border: 1px solid #f0ca70;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 22px;
}


/* Sidebar */
[data-testid="stSidebar"] {
    background: #eef2f8;
}

.sidebar-title {
    font-size: 23px;
    font-weight: 800;
    color: #16213e;
}


/* Footer */
.footer {
    text-align: center;
    color: #7a8496;
    font-size: 13px;
    margin-top: 35px;
    padding-top: 20px;
    border-top: 1px solid #dfe5ef;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🛡️ Screening System</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("System Modules")

    st.write("📄  Document Upload")
    st.write("🔤  OCR Extraction")
    st.write("✓  Document Validation")
    st.write("🔎  Tampering Detection")
    st.write("👤  Face Verification")
    st.write("🎯  Risk Engine")

    st.markdown("---")

    st.success("Main Web Application Ready")

    st.info(
        "Current analysis modules are running in demonstration mode."
    )

    st.markdown("---")

    st.caption("Smart India Hackathon 2026")
    st.caption("Problem Statement: SIH26188")


# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        🛡️ AI-Powered Fake Identity & Document Screening System
    </div>

    <div class="hero-subtitle">
        Smart India Hackathon 2026 • Problem Statement SIH26188
    </div>

    <div class="hero-description">
        An intelligent screening dashboard designed to assist in
        identifying potentially fraudulent identity documents using
        OCR, document validation, tampering analysis and optional
        face verification.
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# DEMONSTRATION NOTICE
# ============================================================

st.markdown("""
<div class="demo-box">

<b>⚠️ DEMONSTRATION MODE</b>

<br><br>

The connected OCR, validation, tampering, face verification and
risk-engine modules currently return placeholder demonstration
results. They are not real identity-verification results yet.

<br><br>

For testing, use fictional/mock documents rather than real personal
identity documents.

</div>
""", unsafe_allow_html=True)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📥 Document Input</div>',
    unsafe_allow_html=True
)

upload_col1, upload_col2 = st.columns(2)

with upload_col1:

    st.markdown(
        '<div class="upload-card">',
        unsafe_allow_html=True
    )

    st.subheader("📄 Identity Document")

    uploaded_document = st.file_uploader(
        "Upload a document image",
        type=["jpg", "jpeg", "png", "webp"],
        key="document_upload"
    )

    st.markdown("</div>", unsafe_allow_html=True)


with upload_col2:

    st.markdown(
        '<div class="upload-card">',
        unsafe_allow_html=True
    )

    st.subheader("👤 Face Photograph")

    uploaded_face = st.file_uploader(
        "Optional photograph for face verification",
        type=["jpg", "jpeg", "png", "webp"],
        key="face_upload"
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("")

analyze = st.button(
    "🔍  ANALYZE DOCUMENT",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if uploaded_document is None:

        st.error(
            "⚠️ Please upload a document before starting the analysis."
        )

    else:

        # ----------------------------------------------------
        # READ DOCUMENT
        # ----------------------------------------------------

        try:

            document_image = Image.open(uploaded_document)
            document_image.load()

        except Exception as error:

            st.error(
                f"❌ Could not read the uploaded document: {error}"
            )
            st.stop()


        # ----------------------------------------------------
        # DOCUMENT PREVIEW
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🖼️ Document Preview</div>',
            unsafe_allow_html=True
        )

        preview_col1, preview_col2 = st.columns([1, 1])

        with preview_col1:

            st.image(
                document_image,
                caption="Uploaded document",
                use_container_width=True
            )

        with preview_col2:

            st.info(
                "Document received successfully.\n\n"
                "The analysis pipeline will now process the "
                "document through the connected modules."
            )


        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        with st.spinner("🔤 Running OCR extraction..."):

            try:

                ocr_result = extract_text(document_image)

            except Exception as error:

                st.error(f"❌ OCR module error: {error}")
                st.stop()


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        with st.spinner("✓ Checking document validation..."):

            try:
                 validation_result = validate_document(
                 ocr_result,
                 document_image
                  )
            except Exception as error:

                st.error(f"❌ Validation module error: {error}")
                st.stop()


        # ----------------------------------------------------
        # TAMPERING
        # ----------------------------------------------------

        with st.spinner("🔎 Checking for possible tampering..."):

            try:

                tamper_result = detect_tampering(
                    document_image
                )

            except Exception as error:

                st.error(f"❌ Tampering module error: {error}")
                st.stop()


        # ----------------------------------------------------
        # FACE VERIFICATION
        # ----------------------------------------------------

        face_image = None

        if uploaded_face is not None:

            try:

                face_image = Image.open(uploaded_face)
                face_image.load()

            except Exception:

                st.warning(
                    "⚠️ Face photograph could not be read. "
                    "Continuing without face verification."
                )

                face_image = None


        with st.spinner("👤 Running face verification..."):

            try:

                face_result = verify_face(
                    document_image,
                    face_image
                )

            except Exception as error:

                st.warning(
                    f"⚠️ Face verification module error: {error}"
                )

                face_result = {
                    "status": "ERROR",
                    "similarity": 0,
                    "message": "Face verification unavailable."
                }


        # ----------------------------------------------------
        # RISK ENGINE
        # ----------------------------------------------------

        with st.spinner("🎯 Calculating overall risk..."):

            try:

                risk_result = calculate_risk(
                    ocr_result,
                    validation_result,
                    tamper_result,
                    face_result
                )

            except Exception as error:

                st.error(f"❌ Risk engine error: {error}")
                st.stop()


        # ====================================================
        # RESULTS
        # ====================================================

        st.markdown(
            '<div class="section-title">📊 Screening Results</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # EXTRACT VALUES
        # ----------------------------------------------------

        document_type = ocr_result.get(
            "document_type",
            "Not available"
        )

        name = ocr_result.get(
            "name",
            "Not available"
        )

        document_number = ocr_result.get(
            "document_number",
            "Not available"
        )

        dob = ocr_result.get(
            "date_of_birth",
            "Not available"
        )

        address = ocr_result.get(
            "address",
            "Not available"
        )

        confidence = ocr_result.get(
            "confidence",
            0
        )

        validation_status = validation_result.get(
            "status",
            "UNKNOWN"
        )

        tamper_status = tamper_result.get(
            "status",
            "UNKNOWN"
        )

        face_status = face_result.get(
            "status",
            "UNKNOWN"
        )

        similarity = face_result.get(
            "similarity",
            0
        )

        score = risk_result.get(
            "score",
            0
        )

        level = risk_result.get(
            "level",
            "UNKNOWN"
        )

        reason = risk_result.get(
            "reason",
            ""
        )


        # ----------------------------------------------------
        # TOP STATUS CARDS
        # ----------------------------------------------------

        st.markdown("#### Module Status")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown(
                f"""
                <div class="status-card">
                    <div class="status-title">🔤 OCR Confidence</div>
                    <div class="status-value">{confidence}%</div>
                    <div class="status-detail">{document_type}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with c2:

            st.markdown(
                f"""
                <div class="status-card">
                    <div class="status-title">✓ Validation</div>
                    <div class="status-value">{validation_status}</div>
                    <div class="status-detail">Document structure check</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with c3:

            st.markdown(
                f"""
                <div class="status-card">
                    <div class="status-title">🔎 Tampering</div>
                    <div class="status-value">{tamper_status}</div>
                    <div class="status-detail">Manipulation analysis</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with c4:

            st.markdown(
                f"""
                <div class="status-card">
                    <div class="status-title">👤 Face Verification</div>
                    <div class="status-value">{face_status}</div>
                    <div class="status-detail">Similarity: {similarity}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # OCR DETAILS
        # ----------------------------------------------------

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.subheader("🔤 OCR Extraction")

        ocr1, ocr2 = st.columns(2)

        with ocr1:

            st.write(f"**Document Type:** {document_type}")
            st.write(f"**Name:** {name}")
            st.write(f"**Document Number:** {document_number}")

        with ocr2:

            st.write(f"**Date of Birth:** {dob}")
            st.write(f"**Address:** {address}")
            st.write(f"**OCR Confidence:** {confidence}%")

        st.markdown("</div>", unsafe_allow_html=True)


        # ----------------------------------------------------
        # VALIDATION + TAMPERING
        # ----------------------------------------------------

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.subheader("✓ Document Validation")

            if validation_status == "PASS":

                st.success(f"Validation Status: {validation_status}")

            else:

                st.warning(
                    f"Validation Status: {validation_status}"
                )

            st.write(
                validation_result.get(
                    "message",
                    "No validation message."
                )
            )

            st.markdown("</div>", unsafe_allow_html=True)


       with result_col2:

           st.markdown(
             '<div class="result-card">',
        unsafe_allow_html=True
    )

    st.subheader("🔎 Tampering Detection")

    # Get tampering score
    tamper_score = tamper_result.get("score", 0)
    
    if "LOW" in str(tamper_status).upper():
        st.success(f"Tampering Status: {tamper_status}")
    else:
        st.warning(f"Tampering Status: {tamper_status}")

    # Display score
    st.metric("Tampering Score", f"{tamper_score}/100")

    # Display message
    st.write(tamper_result.get("message", "No tampering message."))

    # Display detailed scores
    detailed_scores = tamper_result.get("detailed_scores", {})
    if detailed_scores:
        st.write("**Detailed Analysis:**")
        for method, score in detailed_scores.items():
            st.write(f"  • {method}: {score}")

    st.markdown("</div>", unsafe_allow_html=True)


        # ----------------------------------------------------
        # FACE VERIFICATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.subheader("👤 Face Verification")

        face_col1, face_col2 = st.columns(2)

        with face_col1:

            st.metric(
                "Verification Status",
                face_status
            )

        with face_col2:

            st.metric(
                "Similarity",
                f"{similarity}%"
            )

        st.write(
            face_result.get(
                "message",
                "No face verification message."
            )
        )

        st.markdown("</div>", unsafe_allow_html=True)


        # ----------------------------------------------------
        # OVERALL RISK
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🎯 Overall Risk Assessment</div>',
            unsafe_allow_html=True
        )

        risk_col1, risk_col2 = st.columns([1, 2])

        with risk_col1:

            st.markdown(
                f"""
                <div class="risk-card">

                    <div class="risk-label">
                        OVERALL RISK SCORE
                    </div>

                    <div class="risk-score">
                        {score}/100
                    </div>

                    <div class="risk-label">
                        Risk Classification
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with risk_col2:

            st.markdown("### Risk Classification")

            if level == "LOW":

                st.success(
                    f"🟢 LOW RISK — Score: {score}/100"
                )

            elif level == "MEDIUM":

                st.warning(
                    f"🟡 MEDIUM RISK — Score: {score}/100"
                )

            else:

                st.error(
                    f"🔴 HIGH RISK — Score: {score}/100"
                )

            st.progress(
                max(0, min(100, int(score)))
            )

            st.write("**Assessment Reason:**")
            st.write(reason)


        # ----------------------------------------------------
        # INTEGRATED PIPELINE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🔗 Integrated Analysis Pipeline</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="pipeline-card">',
            unsafe_allow_html=True
        )

        p1, p2, p3, p4, p5 = st.columns(5)

        with p1:
            st.success("📄 OCR\n\n✓ Complete")

        with p2:
            st.success("✓ Validation\n\n✓ Complete")

        with p3:
            st.success("🔎 Tampering\n\n✓ Complete")

        with p4:
            st.success("👤 Face\n\n✓ Complete")

        with p5:
            st.success("🎯 Risk Engine\n\n✓ Complete")

        st.markdown("</div>", unsafe_allow_html=True)


        # ----------------------------------------------------
        # INTEGRATION MESSAGE
        # ----------------------------------------------------

        st.info(
            "🔌 Prototype integration successful. "
            "Individual modules can now be replaced with their "
            "actual AI implementations without redesigning the "
            "main application."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        AI-Powered Fake Identity & Document Screening System
        <br>
        Smart India Hackathon 2026 • SIH26188
        <br>
        Prototype Demonstration
    </div>
    """,
    unsafe_allow_html=True
)
