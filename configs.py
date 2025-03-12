
import torch

class LocalConfig:
    # Models
    IMAGE_MODEL = "google/vit-base-patch16-224-in21k"
    TEXT_MODEL = "distilroberta-base"
    PROJ_DIM = 256

    # Training
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 5e-5
    WARMUP = 500
    GRAD_ACCUM = 1

    # System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FP16 = True
    # Save checkpoints to a local folder called "checkpoints"
    SAVE_PATH = "./checkpoints/"

    print(f"Using device: {DEVICE}")

    # Data
    MAX_SEQ_LEN = 64
    IMG_SIZE = 224

