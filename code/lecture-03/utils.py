from PIL import Image

import torch
from torchvision import transforms as tfms

from constants import CUDA_DEVICE



def image_grid(images, rows, cols):
    w,h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(images): grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def encode_image(image, model):
    with torch.no_grad():
        latents = model.encode(tfms.ToTensor()(image).unsqueeze(0).to(CUDA_DEVICE)*2-1)
    return 0.18215 * latents.latent_dist.sample()

def decode_image(latents, model):        
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = model.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")    
    return [Image.fromarray(image) for image in images]

def encode_text(prompts, tokenizer, text_encoder, max_length=None):
    if max_length is None:
        max_length = tokenizer.model_max_length    
    text_input = tokenizer(prompts, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(CUDA_DEVICE))[0]
    return text_embeddings
    