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
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")  # backend sin GUI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.utils import custom_object_scope
import tensorflow as tf
from scipy.ndimage import gaussian_filter

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

# =========
# Configuración global
# =========
PROJECT_ROOT     = Path(".")
RUTA_FONDO       = PROJECT_ROOT / "roca/roca.jpg"
MODEL_PATH       = PROJECT_ROOT / "modelo/mejor_modelo_por_loss_17.h5"
MODEL_PATH_2     = PROJECT_ROOT / "modelo/mejor_modelo.h5"
IMG_SIZE         = 256
MASK_THRESHOLD   = 0.6
MASK_THRESHOLD_2 = 0.75
OVERLAY_ALPHA    = 0.3

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

# =========
# Utilidades
# =========
def img_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def aplicar_tono_color_pil(fondo_pil, mascara_np, intensidad=0.5, color_objetivo=(0.25, 0.3, 0.75)):
    fondo_np = np.array(fondo_pil).astype(float) / 255.0
    fondo_bgr = fondo_np[..., ::-1]

    alpha = mascara_np.astype(float) / 255.0
    alpha = alpha[..., np.newaxis]

    b_obj, g_obj, r_obj = color_objetivo
    B, G, R = fondo_bgr[..., 0], fondo_bgr[..., 1], fondo_bgr[..., 2]

    B_mod = np.clip(B + alpha[..., 0] * intensidad * (b_obj - B), 0, 1)
    G_mod = np.clip(G + alpha[..., 0] * intensidad * (g_obj - G), 0, 1)
    R_mod = np.clip(R + alpha[..., 0] * intensidad * (r_obj - R), 0, 1)

    resultado_bgr = np.stack([B_mod, G_mod, R_mod], axis=-1)
    resultado_rgb = resultado_bgr[..., ::-1]
    resultado_uint8 = (resultado_rgb * 255).astype(np.uint8)

    return PIL.Image.fromarray(resultado_uint8)

def read_image_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def dice_coefficient(y_true, y_pred, smooth=1):
    """Métrica Dice (para cargar modelos que la tengan guardada)."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def ruido_perlin_simple(shape, escala=50, octavas=4, seed=42):
    """Aproximación de ruido Perlin usando sumas de gaussianas."""
    rng = np.random.default_rng(seed)
    ruido = np.zeros(shape, dtype=np.float32)
    amplitud = 1.0
    frecuencia = 1.0
    for _ in range(octavas):
        capa = rng.standard_normal(shape).astype(np.float32)
        sigma = escala / frecuencia
        capa = gaussian_filter(capa, sigma=sigma)
        ruido += amplitud * capa
        amplitud  *= 0.5
        frecuencia *= 2.0
    # Normalizar a [0, 1]
    ruido -= ruido.min()
    ruido /= ruido.max() + 1e-8
    return ruido

def simular_desgaste_poroso(resultado_pil, mascara_np, fondo_pil,
                             intensidad_desgaste=0.45,
                             intensidad_porosidad=0.3,
                             intensidad_rugosidad=0.25,
                             seed=42):
    """
    Aplica sobre la región de la máscara:
      1. Desgaste en bordes  – erosiona el color hacia el fondo en la periferia.
      2. Porosidad           – agujeros pequeños donde asoma el fondo de roca.
      3. Rugosidad/textura   – variación de brillo que imita la superficie irregular.
      4. Micro-desgaste      – manchas de desgaste distribuidas por toda la zona.
    """
    resultado_np = np.array(resultado_pil).astype(np.float32)
    fondo_np     = np.array(fondo_pil).astype(np.float32)
    h, w         = resultado_np.shape[:2]

    mascara_bin = (mascara_np > 127).astype(np.float32)   # 0.0 / 1.0

    # ── 1. DESGASTE EN BORDES ─────────────────────────────────────────────
    # Distancia al borde de la máscara: píxeles cercanos al contorno = más desgaste
    from scipy.ndimage import distance_transform_edt
    dist_interior = distance_transform_edt(mascara_bin)
    dist_max      = dist_interior.max() + 1e-8
    dist_norm     = dist_interior / dist_max              # 0 en borde, 1 en centro

    # Ruido que modula el desgaste (borde irregular, no uniforme)
    ruido_borde = ruido_perlin_simple((h, w), escala=30, octavas=3, seed=seed)
    peso_borde  = np.clip(1.0 - dist_norm * 2.5 + ruido_borde * 0.4, 0, 1)
    peso_borde *= mascara_bin * intensidad_desgaste

    # ── 2. POROSIDAD ──────────────────────────────────────────────────────
    # Puntos aleatorios donde el fondo "asoma" a través del material
    ruido_poros = ruido_perlin_simple((h, w), escala=8, octavas=2, seed=seed+1)
    umbral_poro = 1.0 - intensidad_porosidad
    poros       = (ruido_poros > umbral_poro).astype(np.float32) * mascara_bin

    # Suavizar bordes de los poros para que no sean cuadrados
    poros_suave = gaussian_filter(poros, sigma=1.2)
    poros_suave = np.clip(poros_suave / (poros_suave.max() + 1e-8), 0, 1)

    # ── 3. RUGOSIDAD / VARIACIÓN DE BRILLO ───────────────────────────────
    ruido_rugoso = ruido_perlin_simple((h, w), escala=15, octavas=4, seed=seed+2)
    # Centrar en 1.0 con variación ±intensidad_rugosidad
    factor_brillo = 1.0 + (ruido_rugoso - 0.5) * 2.0 * intensidad_rugosidad
    factor_brillo = factor_brillo[:, :, np.newaxis]      # broadcast a RGB

    # ── 4. MICRO-DESGASTE (manchas interiores) ────────────────────────────
    ruido_micro = ruido_perlin_simple((h, w), escala=20, octavas=3, seed=seed+3)
    umbral_micro = 1.0 - intensidad_desgaste * 0.6
    micro        = (ruido_micro > umbral_micro).astype(np.float32) * mascara_bin
    micro_suave  = gaussian_filter(micro, sigma=2.5)
    micro_suave  = np.clip(micro_suave / (micro_suave.max() + 1e-8), 0, 1) * 0.6

    # ── COMPOSICIÓN FINAL ─────────────────────────────────────────────────
    alfa_borde = (peso_borde + poros_suave + micro_suave)
    alfa_borde = np.clip(alfa_borde, 0, 1)[:, :, np.newaxis]

    # Mezclar resultado con fondo donde hay desgaste/poros
    composicion = resultado_np * (1.0 - alfa_borde) + fondo_np * alfa_borde

    # Aplicar rugosidad solo dentro de la máscara
    mascara_3d  = mascara_bin[:, :, np.newaxis]
    composicion = composicion * (1.0 - mascara_3d) + \
                  np.clip(composicion * factor_brillo, 0, 255) * mascara_3d

    return PIL.Image.fromarray(composicion.astype(np.uint8))

def read_image_rgb(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

def inferir_y_simular(mdl, batch, img_rgb, threshold, label):
    h, w = img_rgb.shape[:2]

    prob      = mdl.predict(batch, verbose=0)[0, :, :, 0]
    prob_full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_bin  = (prob_full > threshold).astype(np.uint8)
    mask_bw   = (prob > threshold).astype(np.uint8) * 255

    mascara_pil      = PIL.Image.fromarray(mask_bw)
    img_original_pil = PIL.Image.fromarray(img_rgb)
    orig_w, orig_h   = img_original_pil.size

    fondo_pil           = PIL.Image.open(RUTA_FONDO).convert("RGB")
    target_w, target_h = fondo_pil.size
    scale               = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h        = int(round(orig_w * scale)), int(round(orig_h * scale))

    mascara_pil_resized = mascara_pil.resize((new_w, new_h), PIL.Image.LANCZOS)
    lienzo              = PIL.Image.new("L", (target_w, target_h), 0)
    offset_x            = (target_w - new_w) // 2
    offset_y            = (target_h - new_h) // 2
    lienzo.paste(mascara_pil_resized, (offset_x, offset_y))
    mascara_np = np.array(lienzo)

    resultado_pil = aplicar_tono_color_pil(
        fondo_pil      = fondo_pil,
        mascara_np     = mascara_np,
        intensidad     = 0.7,
        color_objetivo = (0.35, 0.30, 0.80)
    )

    resultado_final = simular_desgaste_poroso(
        resultado_pil        = resultado_pil,
        mascara_np           = mascara_np,
        fondo_pil            = fondo_pil,
        intensidad_desgaste  = 0.50,
        intensidad_porosidad = 0.30,
        intensidad_rugosidad = 0.25,
        seed                 = 42
    )

    cobertura = mask_bin.mean() * 100
    print(f"[{label}] Cobertura: {cobertura:.2f}% | Threshold: {threshold}")
    return mask_bin, resultado_final, cobertura

@app.on_event("startup")
async def startup_event():
    global G, model, model_2

    print("Cargando modelo desde modelo/pictos512.pkl ...")
    try:
        G = load_model("modelo/pictos512.pkl")  # <- tu función custom (pickle/torch)
        print("Modelo cargado correctamente.")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")

    for p in [MODEL_PATH, MODEL_PATH_2, RUTA_FONDO]:
        if not Path(p).is_file():
            raise FileNotFoundError(f"No existe el archivo: {p}")

    tf.keras.utils.get_custom_objects()["dice_coefficient"] = dice_coefficient
    model   = keras_load_model(str(MODEL_PATH),   compile=False)  # <- Keras
    model_2 = keras_load_model(str(MODEL_PATH_2), compile=False)  # <- Keras

    print("✅ Modelos Keras cargados correctamente.")

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

@app.post("/comparar")
async def comparar_modelos(imagen: UploadFile = File(...)):
    nombre = imagen.filename or ""
    if not nombre.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen (png, jpg, jpeg, bmp, webp).")

    file_bytes = await imagen.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    try:
        img_rgb = read_image_rgb(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    batch       = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

    mask_1, resultado_1, cob_1 = inferir_y_simular(model,   batch, img_rgb, MASK_THRESHOLD,   "Modelo 1 (loss_17)")
    mask_2, resultado_2, cob_2 = inferir_y_simular(model_2, batch, img_rgb, MASK_THRESHOLD_2, "Modelo 2 (mejor)")

    # ── Figura comparación 2×3 ─────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].imshow(img_rgb);             axes[0, 0].set_title("Imagen original");                     axes[0, 0].axis("off")
    axes[0, 1].imshow(mask_1, cmap="gray"); axes[0, 1].set_title(f"Máscara — Modelo 1 ({cob_1:.1f}%)"); axes[0, 1].axis("off")
    axes[0, 2].imshow(resultado_1);         axes[0, 2].set_title("Simulación — Modelo 1");               axes[0, 2].axis("off")
    axes[1, 0].imshow(img_rgb);             axes[1, 0].set_title("Imagen original");                     axes[1, 0].axis("off")
    axes[1, 1].imshow(mask_2, cmap="gray"); axes[1, 1].set_title(f"Máscara — Modelo 2 ({cob_2:.1f}%)"); axes[1, 1].axis("off")
    axes[1, 2].imshow(resultado_2);         axes[1, 2].set_title("Simulación — Modelo 2");               axes[1, 2].axis("off")
    fig1.suptitle("Comparación de modelos de segmentación", fontsize=15, fontweight="bold")
    plt.tight_layout()
    img_comparacion = fig_to_base64(fig1)

    # ── Imágenes resultado individuales (PIL → base64) ─────────────────────
    def pil_to_base64(img_pil: PIL.Image.Image) -> str:
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # ── Respuesta JSON ─────────────────────────────────────────────────────
    return JSONResponse(content={
        "metricas": {
            "cobertura_modelo_1": round(cob_1, 4),
            "cobertura_modelo_2": round(cob_2, 4),
            "threshold_modelo_1": MASK_THRESHOLD,
            "threshold_modelo_2": MASK_THRESHOLD_2,
        },
        "imagenes": {
            "comparacion":         img_comparacion,
            "simulacion_modelo_1": pil_to_base64(resultado_1),
            "simulacion_modelo_2": pil_to_base64(resultado_2),
        }
    })