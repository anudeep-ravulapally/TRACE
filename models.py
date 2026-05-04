import json
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    """A registered TRACE user.

    Gait enrollment now uses a **dual-angle** schema: a separate 512-d
    embedding is stored for a Left-to-Right walk clip and a Right-to-Left
    walk clip. This mitigates the L→R vs R→L covariate shift we observed at
    inference time. Either column may be ``None`` while the user is being
    progressively enrolled; ``has_gait()`` is true as soon as one is set.

    The face column is unchanged (single averaged ArcFace embedding).
    """

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=True, default="General")

    # Face: JSON 512-d list
    embedding = db.Column(db.Text, nullable=True)

    # Gait: JSON 512-d lists, one per walking direction.
    gait_embedding_lr = db.Column(db.Text, nullable=True)  # Left-to-Right clip
    gait_embedding_rl = db.Column(db.Text, nullable=True)  # Right-to-Left clip

    # ── Face Methods ───────────────────────────────────────────────────
    def set_embedding(self, embedding_list):
        self.embedding = json.dumps(embedding_list)

    def get_embedding(self):
        if self.embedding is None:
            return None
        return json.loads(self.embedding)

    def has_face(self):
        return self.embedding is not None

    # ── Gait Methods (dual-angle) ──────────────────────────────────────
    def set_gait_embedding_lr(self, embedding_list):
        """Store the Left-to-Right walk embedding (or clear it with None)."""
        self.gait_embedding_lr = (
            json.dumps(list(embedding_list)) if embedding_list else None
        )

    def get_gait_embedding_lr(self):
        if self.gait_embedding_lr is None:
            return None
        return json.loads(self.gait_embedding_lr)

    def set_gait_embedding_rl(self, embedding_list):
        """Store the Right-to-Left walk embedding (or clear it with None)."""
        self.gait_embedding_rl = (
            json.dumps(list(embedding_list)) if embedding_list else None
        )

    def get_gait_embedding_rl(self):
        if self.gait_embedding_rl is None:
            return None
        return json.loads(self.gait_embedding_rl)

    def get_gait_embeddings(self):
        """Return all stored gait clip embeddings as a list of 512-d lists.

        Used by ``gait_utils.find_best_gait_match`` which takes the per-user
        max of cosine similarities across whichever clips the user has
        enrolled. Returns ``None`` when neither direction is set so callers
        can skip the user cleanly.
        """
        clips = []
        lr = self.get_gait_embedding_lr()
        if lr:
            clips.append(lr)
        rl = self.get_gait_embedding_rl()
        if rl:
            clips.append(rl)
        return clips or None

    def has_gait(self):
        return (
            self.gait_embedding_lr is not None
            or self.gait_embedding_rl is not None
        )
