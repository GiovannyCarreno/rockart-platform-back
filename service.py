import io
import base64
from typing import List, Optional
import os
import numpy as np
import torch
import PIL.Image
import legacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
import random
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

app = FastAPI()

origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Solo este origen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

G = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Image(BaseModel):
    seed: Optional[float] = 456
    truncation_psi: float = 1.0
    noise_mode: str = 'const'  # 'const', 'random', 'none'
    number: Optional[int] = 1

def load_model(model_path: str):
    global G
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    with open(model_path, 'rb') as f:
        G_local = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    return G_local

@app.on_event("startup")
async def startup_event():
    global G
    model_path = "modelo/pictos512.pkl"  # Ajusta tu ruta aquí
    print(f"Cargando modelo desde {model_path} ...")
    try:
        G = load_model(model_path)
        print("Modelo cargado correctamente.")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")

@app.post("/generateSingle")
async def generate_image(image: Image):
    seed = int(image.seed)
    truncation_psi = image.truncation_psi
    noise_mode = image.noise_mode

    global G
    if G is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    label = torch.zeros([1, G.c_dim], device=device)

    # Generar imágen con GAN
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")

    return {
        "image": img_to_b64(pil_img),
        "seed": seed,
        "truncation_psi": truncation_psi,
        "noise_mode":  noise_mode
    }

@app.post("/generateSeveral")
async def generate_several(image: Image):
    truncation_psi = image.truncation_psi
    noise_mode = image.noise_mode
    
    global G
    if G is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    label = torch.zeros([1, G.c_dim], device=device)

    images = []
    seeds = []

    for i in range(0, image.number, 1):
        seed = random.randint(1, 2147483647)

        # Generar imágen con GAN
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")

        images.append(img_to_b64(pil_img))
        seeds.append(seed)

    return {
        "number": image.number,
        "images": images,
        "seeds": seeds,
        "truncation_psi": truncation_psi,
        "noise_mode":  noise_mode
    }

def img_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()