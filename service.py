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
matplotlib.use("Agg")
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import onnxruntime as ort
import tensorflow as tf
from scipy.ndimage import gaussian_filter, distance_transform_edt

warnings.filterwarnings("ignore")

app = FastAPI()

origins = ["http://localhost:5174"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========
# Configuración global
# =========
PROJECT_ROOT   = Path(".")
RUTA_FONDO     = PROJECT_ROOT / "roca/roca_3.jpg"
ONNX_MODEL_PATH = PROJECT_ROOT / "modelo/mejor_modelo_dinamico.onnx"

RESOLUTIONS    = [256, 512]          # las dos resoluciones a comparar
MASK_THRESHOLD = 0.7                 # mismo threshold para ambas
OVERLAY_ALPHA  = 0.3

G = None
onnx_session = None                  # sesión ONNX compartida
onnx_input_name = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Modelos Keras eliminados ───────────────────────────────────────────────
# model   = None   (ya no se usa)
# model_2 = None   (ya no se usa)

class Image(BaseModel):
    seed: Optional[float] = 456
    truncation_psi: float = 1.0
    noise_mode: str = 'const'
    number: Optional[int] = 1

def load_model(model_path: str):
    global G
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    with open(model_path, 'rb') as f:
        G_local = legacy.load_network_pkl(f)['G_ema'].to(device)
    return G_local

# =========
# Utilidades generales
# =========
def img_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def pil_to_base64(img_pil: PIL.Image.Image) -> str:
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

def read_image_rgb(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =========
# Utilidades de simulación  (sin cambios)
# =========
def ruido_perlin_simple(shape, escala=50, octavas=4, seed=42):
    rng = np.random.default_rng(seed)
    ruido = np.zeros(shape, dtype=np.float32)
    amplitud, frecuencia = 1.0, 1.0
    for _ in range(octavas):
        capa = rng.standard_normal(shape).astype(np.float32)
        capa = gaussian_filter(capa, sigma=escala / frecuencia)
        ruido += amplitud * capa
        amplitud  *= 0.5
        frecuencia *= 2.0
    ruido -= ruido.min()
    ruido /= ruido.max() + 1e-8
    return ruido

def aplicar_tono_color_pil(fondo_pil, mascara_np, intensidad=0.7,
                            color_objetivo=(0.35, 0.30, 0.80)):
    fondo_np  = np.array(fondo_pil).astype(float) / 255.0
    fondo_bgr = fondo_np[..., ::-1]
    alpha     = (mascara_np.astype(float) / 255.0)[..., np.newaxis]
    b_obj, g_obj, r_obj = color_objetivo
    B, G_ch, R = fondo_bgr[..., 0], fondo_bgr[..., 1], fondo_bgr[..., 2]
    B_mod = np.clip(B + alpha[..., 0] * intensidad * (b_obj - B), 0, 1)
    G_mod = np.clip(G_ch + alpha[..., 0] * intensidad * (g_obj - G_ch), 0, 1)
    R_mod = np.clip(R + alpha[..., 0] * intensidad * (r_obj - R), 0, 1)
    resultado_rgb = np.stack([B_mod, G_mod, R_mod], axis=-1)[..., ::-1]
    return PIL.Image.fromarray((resultado_rgb * 255).astype(np.uint8))

def simular_desgaste_poroso(resultado_pil, mascara_np, fondo_pil,
                             intensidad_desgaste=0.50,
                             intensidad_porosidad=0.30,
                             intensidad_rugosidad=0.25,
                             seed=42):
    resultado_np = np.array(resultado_pil).astype(np.float32)
    fondo_np     = np.array(fondo_pil).astype(np.float32)
    h, w         = resultado_np.shape[:2]
    mascara_bin  = (mascara_np > 127).astype(np.float32)

    dist_interior = distance_transform_edt(mascara_bin)
    dist_norm     = dist_interior / (dist_interior.max() + 1e-8)
    ruido_borde   = ruido_perlin_simple((h, w), escala=30, octavas=3, seed=seed)
    peso_borde    = np.clip(1.0 - dist_norm * 2.5 + ruido_borde * 0.4, 0, 1)
    peso_borde   *= mascara_bin * intensidad_desgaste

    ruido_poros = ruido_perlin_simple((h, w), escala=8, octavas=2, seed=seed+1)
    poros       = ((ruido_poros > 1.0 - intensidad_porosidad) * mascara_bin).astype(np.float32)
    poros_suave = gaussian_filter(poros, sigma=1.2)
    poros_suave = np.clip(poros_suave / (poros_suave.max() + 1e-8), 0, 1)

    ruido_rugoso  = ruido_perlin_simple((h, w), escala=15, octavas=4, seed=seed+2)
    factor_brillo = (1.0 + (ruido_rugoso - 0.5) * 2.0 * intensidad_rugosidad)[:, :, np.newaxis]

    ruido_micro = ruido_perlin_simple((h, w), escala=20, octavas=3, seed=seed+3)
    micro       = ((ruido_micro > 1.0 - intensidad_desgaste * 0.6) * mascara_bin).astype(np.float32)
    micro_suave = gaussian_filter(micro, sigma=2.5)
    micro_suave = np.clip(micro_suave / (micro_suave.max() + 1e-8), 0, 1) * 0.6

    alfa       = np.clip(peso_borde + poros_suave + micro_suave, 0, 1)[:, :, np.newaxis]
    composicion = resultado_np * (1 - alfa) + fondo_np * alfa
    mascara_3d  = mascara_bin[:, :, np.newaxis]
    composicion = composicion * (1 - mascara_3d) + \
                  np.clip(composicion * factor_brillo, 0, 255) * mascara_3d
    return PIL.Image.fromarray(composicion.astype(np.uint8))

# =========
# Pipeline ONNX  (reemplaza inferir_y_simular con Keras)
# =========
def inferir_y_simular_onnx(img_rgb: np.ndarray, img_size: int,
                            threshold: float, label: str):
    """
    Misma firma de salida que el inferir_y_simular original:
        mask_bin (np.uint8 H×W), resultado_final (PIL.Image), cobertura (float)
    """
    global onnx_session, onnx_input_name
    h_orig, w_orig = img_rgb.shape[:2]

    # ── Inferencia ONNX ───────────────────────────────────────────────────
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    batch       = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
    output      = onnx_session.run(None, {onnx_input_name: batch})[0]
    prob        = output[0, :, :, 0]                           # (img_size, img_size)

    prob_full = cv2.resize(prob, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    mask_bin  = (prob_full > threshold).astype(np.uint8)
    mask_bw   = (prob > threshold).astype(np.uint8) * 255     # tamaño del modelo

    # ── Escalar máscara al tamaño del fondo ──────────────────────────────
    mascara_pil        = PIL.Image.fromarray(mask_bw)
    fondo_pil          = PIL.Image.open(RUTA_FONDO).convert("RGB")
    target_w, target_h = fondo_pil.size
    scale              = min(target_w / w_orig, target_h / h_orig)
    new_w, new_h       = int(round(w_orig * scale)), int(round(h_orig * scale))
    mascara_resized    = mascara_pil.resize((new_w, new_h), PIL.Image.LANCZOS)
    lienzo             = PIL.Image.new("L", (target_w, target_h), 0)
    lienzo.paste(mascara_resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    mascara_np         = np.array(lienzo)

    # ── Simulación (idéntica al original) ────────────────────────────────
    resultado_pil = aplicar_tono_color_pil(fondo_pil, mascara_np,
                                            intensidad=0.7,
                                            color_objetivo=(0.35, 0.30, 0.80))
    resultado_final = simular_desgaste_poroso(resultado_pil, mascara_np, fondo_pil,
                                               intensidad_desgaste=0.50,
                                               intensidad_porosidad=0.10,
                                               intensidad_rugosidad=0.25,
                                               seed=42)
    cobertura = mask_bin.mean() * 100
    print(f"[{label}] res={img_size} | Cobertura: {cobertura:.2f}% | Threshold: {threshold}")
    return mask_bin, resultado_final, cobertura

# =========
# Startup
# =========
@app.on_event("startup")
async def startup_event():
    global G, onnx_session, onnx_input_name

    # GAN (sin cambios)
    print("Cargando modelo GAN desde modelo/pictos512.pkl ...")
    try:
        G = load_model("modelo/pictos512.pkl")
        print("Modelo GAN cargado correctamente.")
    except Exception as e:
        print(f"Error cargando el modelo GAN: {e}")

    # Validar archivos necesarios
    for p in [ONNX_MODEL_PATH, RUTA_FONDO]:
        if not Path(p).is_file():
            raise FileNotFoundError(f"No existe el archivo: {p}")

    # Sesión ONNX  (GPU si está disponible)
    providers = ort.get_available_providers()
    onnx_session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
    )
    onnx_input_name = onnx_session.get_inputs()[0].name
    print(f"✅ Modelo ONNX cargado | input: '{onnx_input_name}' | "
          f"{'GPU' if 'CUDAExecutionProvider' in providers else 'CPU'}")

# =========
# Endpoints GAN  (sin cambios)
# =========
@app.post("/generateSingle")
async def generate_image(image: Image):
    seed = int(image.seed)
    truncation_psi = image.truncation_psi
    noise_mode = image.noise_mode
    global G
    if G is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    label = torch.zeros([1, G.c_dim], device=device)
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")
    return {"image": img_to_b64(pil_img), "seed": seed,
            "truncation_psi": truncation_psi, "noise_mode": noise_mode}

@app.post("/generateSeveral")
async def generate_several(image: Image):
    truncation_psi = image.truncation_psi
    noise_mode = image.noise_mode
    global G
    if G is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    label = torch.zeros([1, G.c_dim], device=device)
    images, seeds = [], []
    for _ in range(image.number):
        seed = random.randint(1, 2147483647)
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")
        images.append(img_to_b64(pil_img))
        seeds.append(seed)
    return {"number": image.number, "images": images, "seeds": seeds,
            "truncation_psi": truncation_psi, "noise_mode": noise_mode}

# =========
# /comparar  — ahora compara 256×256 vs 512×512 con el modelo ONNX
# =========
@app.post("/comparar")
async def comparar_modelos(imagen: UploadFile = File(...)):
    nombre = imagen.filename or ""
    if not nombre.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        raise HTTPException(status_code=400,
                            detail="El archivo debe ser una imagen (png, jpg, jpeg, bmp, webp).")

    file_bytes = await imagen.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    try:
        img_rgb = read_image_rgb(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ── Inferencia con ambas resoluciones ────────────────────────────────
    mask_1, resultado_1, cob_1 = inferir_y_simular_onnx(
        img_rgb, img_size=256, threshold=MASK_THRESHOLD, label="256×256")
    mask_2, resultado_2, cob_2 = inferir_y_simular_onnx(
        img_rgb, img_size=512, threshold=MASK_THRESHOLD, label="512×512")

    # ── Figura comparación 2×3  (layout idéntico al original) ────────────
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(img_rgb);             axes[0, 0].set_title("Imagen original");                        axes[0, 0].axis("off")
    axes[0, 1].imshow(mask_1, cmap="gray"); axes[0, 1].set_title(f"Máscara — 256×256 ({cob_1:.1f}%)");     axes[0, 1].axis("off")
    axes[0, 2].imshow(resultado_1);         axes[0, 2].set_title("Simulación — 256×256");                   axes[0, 2].axis("off")

    axes[1, 0].imshow(img_rgb);             axes[1, 0].set_title("Imagen original");                        axes[1, 0].axis("off")
    axes[1, 1].imshow(mask_2, cmap="gray"); axes[1, 1].set_title(f"Máscara — 512×512 ({cob_2:.1f}%)");     axes[1, 1].axis("off")
    axes[1, 2].imshow(resultado_2);         axes[1, 2].set_title("Simulación — 512×512");                   axes[1, 2].axis("off")

    fig1.suptitle("Comparación de resoluciones de segmentación (modelo ONNX)",
                  fontsize=15, fontweight="bold")
    plt.tight_layout()
    img_comparacion = fig_to_base64(fig1)

    # ── Respuesta JSON  (mismas claves que el endpoint original) ─────────
    return JSONResponse(content={
        "metricas": {
            "cobertura_modelo_1": round(cob_1, 4),   # resolución 256
            "cobertura_modelo_2": round(cob_2, 4),   # resolución 512
            "threshold_modelo_1": MASK_THRESHOLD,
            "threshold_modelo_2": MASK_THRESHOLD,
        },
        "imagenes": {
            "comparacion":         img_comparacion,
            "simulacion_modelo_1": pil_to_base64(resultado_1),
            "simulacion_modelo_2": pil_to_base64(resultado_2),
        }
    })