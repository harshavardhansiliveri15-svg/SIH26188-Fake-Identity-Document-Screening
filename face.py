from deepface import DeepFace
from PIL import Image
import tempfile
import os


def save_image_as_temp(image):

    if isinstance(image, Image.Image):

        temp = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        )

        image.convert("RGB").save(
            temp.name,
            format="JPEG"
        )

        temp.close()

        return temp.name

    if isinstance(image, str):
        return image

    raise ValueError("Unsupported image format")


def verify_face(document_image, face_image):

    if document_image is None or face_image is None:

        return {
            "status": "NOT PROVIDED",
            "similarity": 0,
            "message": "A document image and face photograph are required."
        }

    temp_document = None
    temp_face = None

    try:

        temp_document = save_image_as_temp(document_image)
        temp_face = save_image_as_temp(face_image)

        result = DeepFace.verify(
            img1_path=temp_document,
            img2_path=temp_face,
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=True
        )

        verified = bool(result["verified"])
        distance = float(result["distance"])

        similarity = max(
            0,
            min(100, (1 - distance) * 100)
        )

        if verified:

            return {
                "status": "MATCH",
                "similarity": round(similarity, 2),
                "message": (
                    "The face comparison indicates a match "
                    "in this prototype analysis."
                )
            }

        return {
            "status": "MISMATCH",
            "similarity": round(similarity, 2),
            "message": (
                "The face comparison indicates that the two "
                "faces do not match in this prototype analysis."
            )
        }

    except Exception as error:

        print("FACE VERIFICATION ERROR:", repr(error))

        return {
            "status": "UNAVAILABLE",
            "similarity": 0,
            "message": (
                "Face verification could not be completed. "
                "Ensure both images contain a clearly detectable face."
            )
        }

    finally:

        for path in [temp_document, temp_face]:

            if path and os.path.exists(path):

                try:
                    os.remove(path)
                except Exception:
                    pass