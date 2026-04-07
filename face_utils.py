# face_utils.py
import numpy as np
from deepface import DeepFace
import base64, cv2

MODEL_NAME = "ArcFace"
THRESHOLD  = 0.75   # Cosine similarity threshold (tune this later)

def base64_to_image(b64_string):
    """Convert a base64 webcam capture to an OpenCV image."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def get_embedding(image):
    """Extract a 512-number ArcFace embedding from an image."""
    result = DeepFace.represent(
        img_path=image,
        model_name=MODEL_NAME,
        enforce_detection=True,   # Raises error if no face found
        detector_backend="mtcnn"
    )
    return result[0]["embedding"]  # Returns list of 512 floats

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