# face_utils.py
import numpy as np
from deepface import DeepFace
import base64, cv2

MODEL_NAME = "ArcFace"
THRESHOLD  = 0.68   # Optimal cosine similarity threshold for ArcFace + RetinaFace

def base64_to_image(b64_string):
    """Convert a base64 webcam capture to an OpenCV image."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def get_embedding(image):
    """Extract a 512-number ArcFace embedding from an image using RetinaFace detector."""
    result = DeepFace.represent(
        img_path=image,
        model_name=MODEL_NAME,
        enforce_detection=True,
        detector_backend="retinaface"   # More robust than mtcnn for contorted faces
    )
    return result[0]["embedding"]  # Returns list of 512 floats

def get_averaged_embedding(images_b64):
    """
    Given a list of base64 image strings, extract ArcFace embeddings for each
    valid face and return the mean (averaged) embedding vector.

    Frames where no face is detected are silently skipped.
    Raises ValueError if no valid face was found in any frame.
    """
    embeddings = []
    for b64 in images_b64:
        try:
            image = base64_to_image(b64)
            emb   = get_embedding(image)
            embeddings.append(emb)
        except Exception:
            # Skip frames where detection failed (no face, blur, etc.)
            continue

    if not embeddings:
        raise ValueError(
            "No face could be detected in any of the captured frames. "
            "Please ensure your face is clearly visible and try again."
        )

    # Stack into (N, 512) matrix and compute column-wise mean → (512,) vector
    mean_embedding = np.mean(np.array(embeddings), axis=0).tolist()
    return mean_embedding

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two embedding vectors."""
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def find_best_match(new_embedding, all_users):
    """Compare new embedding against all stored users. Return best match."""
    best_user  = None
    best_score = -1

    for user in all_users:
        stored_embedding = user.get_embedding()
        score = cosine_similarity(new_embedding, stored_embedding)
        if score > best_score:
            best_score = score
            best_user  = user

    if best_score >= THRESHOLD:
        return best_user, best_score
    return None, best_score  # No match found