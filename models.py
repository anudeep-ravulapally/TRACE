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
    def set_gait_embedding(self, embedding_list):
        self.gait_embedding = json.dumps(embedding_list)

    def get_gait_embedding(self):
        if self.gait_embedding is None:
            return None
        return json.loads(self.gait_embedding)
        
    def has_gait(self):
        return self.gait_embedding is not None