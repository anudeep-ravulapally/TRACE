import json
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=True, default="General")
    embedding = db.Column(db.Text, nullable=True)      # Face: JSON 512-d
    gait_embedding = db.Column(db.Text, nullable=True) # Gait: JSON 512-d

    # --- Face Methods ---
    def set_embedding(self, embedding_list):
        self.embedding = json.dumps(embedding_list)

    def get_embedding(self):
        if self.embedding is None:
            return None
        return json.loads(self.embedding)
        
    def has_face(self):
        return self.embedding is not None

    # --- Gait Methods ---
    # The ``gait_embedding`` text column stores JSON in one of two shapes:
    #   - a flat list of floats (single-clip, legacy)
    #   - a list of flat lists (multi-clip enrollment, new)
    # Both shapes are accepted by readers; writers persist whichever shape the
    # caller asks for. ``get_gait_embeddings()`` (plural) always returns the
    # multi-clip view, while ``get_gait_embedding()`` (singular) returns the
    # mean of all clips for backward compatibility with code that expects one
    # vector per user.

    @staticmethod
    def _is_multi_clip(parsed):
        """Detect whether parsed JSON is a list-of-lists (multi-clip)."""
        return (
            isinstance(parsed, list)
            and len(parsed) > 0
            and isinstance(parsed[0], list)
        )

    def set_gait_embedding(self, embedding_list):
        """Store a single 512-d embedding (legacy single-clip API)."""
        self.gait_embedding = json.dumps(embedding_list)

    def set_gait_embeddings(self, embedding_lists):
        """Store multiple clip embeddings for this user (multi-clip enrollment).

        ``embedding_lists`` is an iterable of 512-d lists. An empty input
        clears the stored embedding.
        """
        clips = [list(e) for e in embedding_lists if e]
        if not clips:
            self.gait_embedding = None
            return
        self.gait_embedding = json.dumps(clips)

    def add_gait_embedding(self, embedding_list):
        """Append a clip embedding without losing previously enrolled ones."""
        existing = self.get_gait_embeddings() or []
        existing.append(list(embedding_list))
        self.set_gait_embeddings(existing)

    def get_gait_embedding(self):
        """Return the user's gait embedding as a single 512-d list.

        For multi-clip storage, returns the (unnormalized) **mean** of all
        stored clips so callers that expect one vector still work.
        """
        if self.gait_embedding is None:
            return None
        parsed = json.loads(self.gait_embedding)
        if not self._is_multi_clip(parsed):
            return parsed
        # Multi-clip → mean across clips. We deliberately don't L2-normalize
        # here because gait_utils handles normalization based on model config.
        n = len(parsed)
        if n == 0:
            return None
        dim = len(parsed[0])
        out = [0.0] * dim
        for clip in parsed:
            for i, v in enumerate(clip):
                out[i] += v
        return [v / n for v in out]

    def get_gait_embeddings(self):
        """Return all stored clip embeddings as a list of 512-d lists.

        Always returns a list (possibly with one element). ``None`` only when
        no embedding has been stored.
        """
        if self.gait_embedding is None:
            return None
        parsed = json.loads(self.gait_embedding)
        if self._is_multi_clip(parsed):
            return parsed
        return [parsed]

    def has_gait(self):
        return self.gait_embedding is not None