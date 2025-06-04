import os
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer

def save_checkpoint(model: SentenceTransformer, checkpoint_dir: str, metadata: dict):
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model
    model.save(checkpoint_dir)

    # Add timestamp
    metadata["timestamp"] = datetime.now().isoformat()

    # Save log
    log_path = os.path.join(checkpoint_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Model and log saved at: {checkpoint_dir}")
