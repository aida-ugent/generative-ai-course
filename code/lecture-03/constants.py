from pathlib import Path

UNCONDITIONAL_DIFFUSION_MODEL_REPO = "google/ddpm-church-256"
DIFFUSION_MODEL_REPO = "runwayml/stable-diffusion-v1-5"
CLIP_MODEL_REPO = "openai/clip-vit-large-patch14"
TMP_FOLDER_PATH = Path("./tmp")
TMP_FOLDER_PATH.mkdir(exist_ok=True)
MODEL_FOLDER_PATH = TMP_FOLDER_PATH / "models"
MODEL_FOLDER_PATH.mkdir(exist_ok=True, parents=True)
IMAGE_FOLDER_PATH = TMP_FOLDER_PATH / "images"
IMAGE_FOLDER_PATH.mkdir(exist_ok=True, parents=True)
CUDA_DEVICE = "cuda:2"

TEXT_TO_IMAGE_PROMPTS = ["An image of church in Ghent", "An image of church in Cologne", "An image of church in Barcelona"]