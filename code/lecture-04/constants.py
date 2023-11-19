from pathlib import Path

TMP_FOLDER_PATH = Path("./tmp")
TMP_FOLDER_PATH.mkdir(exist_ok=True)
MODEL_FOLDER_PATH = TMP_FOLDER_PATH / "models"
MODEL_FOLDER_PATH.mkdir(exist_ok=True, parents=True)
DATA_FOLDER_PATH = TMP_FOLDER_PATH / "data"
DATA_FOLDER_PATH.mkdir(exist_ok=True, parents=True)

XTTS_MODEL_REPO = "tts_models/multilingual/multi-dataset/xtts_v2"
CUDA_DEVICE = "cuda:2"
