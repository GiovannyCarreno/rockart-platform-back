import io
import base64
from typing import List, Literal, Optional
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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import onnxruntime as ort
import tensorflow as tf
from scipy.ndimage import gaussian_filter, distance_transform_edt
import segmentar_petroglifo as petroglyph
import timm
from torchvision import transforms

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
RUTA_FONDO_GAB = PROJECT_ROOT / "roca/roca_5.jpg"

ONNX_MODEL_PATHS = {
    "mejor_modelo_dinamico": PROJECT_ROOT / "modelo/mejor_modelo_dinamico.onnx",
    "modelo_dinamico_gab": PROJECT_ROOT / "modelo/modelo_dinamico_gab.onnx",
}
OnnxModelName = Literal["mejor_modelo_dinamico", "modelo_dinamico_gab"]

RESOLUTIONS    = [256, 512]          # las dos resoluciones a comparar
MASK_THRESHOLD = 0.7                 # mejor_modelo_dinamico
GAB_MASK_THRESHOLD = petroglyph.DEFAULT_THRESHOLD
GAB_MIN_AREA = petroglyph.DEFAULT_MIN_AREA
OVERLAY_ALPHA  = 0.3

MODEL_PATHS = {
    "pictos512": PROJECT_ROOT / "modelo/pictos512.pkl",
    "pictos512_2": PROJECT_ROOT / "modelo/pictos512_2.pkl",
}
G_models: dict[str, torch.nn.Module | None] = {name: None for name in MODEL_PATHS}
onnx_sessions: dict[str, ort.InferenceSession | None] = {name: None for name in ONNX_MODEL_PATHS}
onnx_input_names: dict[str, str | None] = {name: None for name in ONNX_MODEL_PATHS}

CLASSIFIER_MODEL_PATH = PROJECT_ROOT / "modelo/best_model_fine_mobilenet.pth"
CLASSIFIER_IMG_SIZE = 256
CLASS_NAMES = ["Petroglifo", "Pictograma"]
classifier_model: torch.nn.Module | None = None
classifier_transform = transforms.Compose([
    transforms.Resize((CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
    model: Literal["pictos512", "pictos512_2"] = "pictos512"

def load_model(model_path: Path) -> torch.nn.Module:
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    with open(model_path, 'rb') as f:
        return legacy.load_network_pkl(f)['G_ema'].to(device)

def get_generator(model_name: str) -> torch.nn.Module:
    if model_name not in MODEL_PATHS:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo no válido. Use uno de: {list(MODEL_PATHS.keys())}",
        )
    G = G_models.get(model_name)
    if G is None:
        raise HTTPException(
            status_code=500,
            detail=f"Modelo '{model_name}' no cargado",
        )
    return G

def build_onnx_providers() -> list[str | tuple[str, dict]]:
    """Prioridad: NVIDIA (CUDA) → GPU Windows (DirectML) → CPU."""
    available = set(ort.get_available_providers())
    providers: list[str | tuple[str, dict]] = []
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    elif "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def load_onnx_session(model_path: Path) -> tuple[ort.InferenceSession, str, str]:
    session = ort.InferenceSession(str(model_path), providers=build_onnx_providers())
    return session, session.get_inputs()[0].name, session.get_providers()[0]

def load_classifier(model_path: Path) -> torch.nn.Module:
    model = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=False,
        num_classes=len(CLASS_NAMES),
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def classify_image(img: PIL.Image.Image) -> tuple[str, float, dict[str, float]]:
    if classifier_model is None:
        raise HTTPException(status_code=500, detail="Modelo de clasificación no cargado")
    tensor = classifier_transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classifier_model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()
    idx = int(torch.argmax(probs).item())
    clase = CLASS_NAMES[idx]
    confianza = float(probs[idx].item())
    probabilidades = {
        CLASS_NAMES[i]: round(float(probs[i].item()), 4) for i in range(len(CLASS_NAMES))
    }
    return clase, confianza, probabilidades


def get_onnx_session(model_name: str) -> tuple[ort.InferenceSession, str]:
    if model_name not in ONNX_MODEL_PATHS:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo ONNX no válido. Use uno de: {list(ONNX_MODEL_PATHS.keys())}",
        )
    session = onnx_sessions.get(model_name)
    input_name = onnx_input_names.get(model_name)
    if session is None or input_name is None:
        raise HTTPException(
            status_code=500,
            detail=f"Modelo ONNX '{model_name}' no cargado",
        )
    return session, input_name

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
# Pipeline ONNX
# =========
def predict_onnx_tta(
    session: ort.InferenceSession,
    input_name: str,
    img_pre: np.ndarray,
) -> np.ndarray:
    def pred(x: np.ndarray) -> np.ndarray:
        batch = np.expand_dims(x.astype(np.float32), axis=0)
        return session.run(None, {input_name: batch})[0][0, :, :, 0]

    return np.mean([
        pred(img_pre),
        np.fliplr(pred(np.fliplr(img_pre))),
        np.flipud(pred(np.flipud(img_pre))),
        np.fliplr(np.flipud(pred(np.fliplr(np.flipud(img_pre))))),
    ], axis=0)


def paste_on_fondo(
    rendered_rgb: np.ndarray,
    w_orig: int,
    h_orig: int,
    ruta_fondo: Path = RUTA_FONDO,
) -> PIL.Image.Image:
    fondo_pil = PIL.Image.open(ruta_fondo).convert("RGB")
    target_w, target_h = fondo_pil.size
    scale = min(target_w / w_orig, target_h / h_orig)
    new_w, new_h = int(round(w_orig * scale)), int(round(h_orig * scale))
    rendered_pil = PIL.Image.fromarray(rendered_rgb).resize((new_w, new_h), PIL.Image.LANCZOS)
    lienzo = fondo_pil.copy()
    lienzo.paste(rendered_pil, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return lienzo


def inferir_mejor_modelo_onnx(
    img_rgb: np.ndarray,
    img_size: int,
    threshold: float,
    label: str,
    model_name: str,
) -> tuple[np.ndarray, PIL.Image.Image, float]:
    onnx_session, onnx_input_name = get_onnx_session(model_name)
    h_orig, w_orig = img_rgb.shape[:2]

    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    batch = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
    output = onnx_session.run(None, {onnx_input_name: batch})[0]
    prob = output[0, :, :, 0]

    prob_full = cv2.resize(prob, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    mask_bin = (prob_full > threshold).astype(np.uint8)
    mask_bw = (prob > threshold).astype(np.uint8) * 255

    mascara_pil = PIL.Image.fromarray(mask_bw)
    fondo_pil = PIL.Image.open(RUTA_FONDO).convert("RGB")
    target_w, target_h = fondo_pil.size
    scale = min(target_w / w_orig, target_h / h_orig)
    new_w, new_h = int(round(w_orig * scale)), int(round(h_orig * scale))
    mascara_resized = mascara_pil.resize((new_w, new_h), PIL.Image.LANCZOS)
    lienzo = PIL.Image.new("L", (target_w, target_h), 0)
    lienzo.paste(mascara_resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    mascara_np = np.array(lienzo)

    resultado_pil = aplicar_tono_color_pil(
        fondo_pil, mascara_np, intensidad=0.7, color_objetivo=(0.35, 0.30, 0.80),
    )
    resultado_final = simular_desgaste_poroso(
        resultado_pil, mascara_np, fondo_pil,
        intensidad_desgaste=0.50,
        intensidad_porosidad=0.10,
        intensidad_rugosidad=0.25,
        seed=42,
    )
    cobertura = mask_bin.mean() * 100
    print(f"[{label}] res={img_size} | Cobertura: {cobertura:.2f}% | Threshold: {threshold}")
    return mask_bin, resultado_final, cobertura


def inferir_gab_onnx(
    img_rgb: np.ndarray,
    img_size: int,
    threshold: float,
    label: str,
    model_name: str,
) -> tuple[np.ndarray, PIL.Image.Image, float]:
    onnx_session, onnx_input_name = get_onnx_session(model_name)
    h_orig, w_orig = img_rgb.shape[:2]

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_pre = petroglyph.preprocess(img_bgr, img_size)
    probability = predict_onnx_tta(onnx_session, onnx_input_name, img_pre)

    selected_mask = petroglyph.select_best_mask(probability, threshold, GAB_MIN_AREA)
    filled_mask = selected_mask["mask"]
    metrics = selected_mask["metrics"]

    prob_full = cv2.resize(probability, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    mask_full = cv2.resize(filled_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask_full > 127).astype(np.uint8)

    background_rgb, _ = petroglyph.choose_background(RUTA_FONDO_GAB, img_size)
    rendered_rgb = petroglyph.render_petroglyph(background_rgb, filled_mask)
    resultado_final = paste_on_fondo(rendered_rgb, w_orig, h_orig, RUTA_FONDO_GAB)

    cobertura = float(metrics["area_percent"])
    print(
        f"[{label}] res={img_size} | Cobertura: {cobertura:.2f}% | "
        f"Threshold: {selected_mask['threshold']} | Estrategia: {selected_mask['strategy']}"
    )
    return mask_bin, resultado_final, cobertura


def inferir_y_simular_onnx(
    img_rgb: np.ndarray,
    img_size: int,
    threshold: float,
    label: str,
    model_name: str,
) -> tuple[np.ndarray, PIL.Image.Image, float]:
    if model_name == "modelo_dinamico_gab":
        return inferir_gab_onnx(img_rgb, img_size, threshold, label, model_name)
    return inferir_mejor_modelo_onnx(img_rgb, img_size, threshold, label, model_name)

# =========
# Startup
# =========
@app.on_event("startup")
async def startup_event():
    global G_models, onnx_sessions, onnx_input_names, classifier_model

    for name, path in MODEL_PATHS.items():
        print(f"Cargando modelo GAN '{name}' desde {path} ...")
        try:
            G_models[name] = load_model(path)
            print(f"Modelo GAN '{name}' cargado correctamente.")
        except Exception as e:
            print(f"Error cargando el modelo GAN '{name}': {e}")

    for ruta_fondo in (RUTA_FONDO, RUTA_FONDO_GAB):
        if not ruta_fondo.is_file():
            raise FileNotFoundError(f"No existe el archivo: {ruta_fondo}")

    print(f"ONNX Runtime — proveedores disponibles: {ort.get_available_providers()}")
    for name, path in ONNX_MODEL_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"No existe el archivo: {path}")
        print(f"Cargando modelo ONNX '{name}' desde {path} ...")
        try:
            session, input_name, active_provider = load_onnx_session(path)
            onnx_sessions[name] = session
            onnx_input_names[name] = input_name
            print(
                f"Modelo ONNX '{name}' cargado | input: '{input_name}' | "
                f"inferencia en: {active_provider}"
            )
        except Exception as e:
            print(f"Error cargando el modelo ONNX '{name}': {e}")

    if not CLASSIFIER_MODEL_PATH.is_file():
        print(f"Advertencia: no existe el modelo de clasificación: {CLASSIFIER_MODEL_PATH}")
    else:
        print(f"Cargando modelo de clasificación desde {CLASSIFIER_MODEL_PATH} ...")
        try:
            classifier_model = load_classifier(CLASSIFIER_MODEL_PATH)
            print(f"Modelo de clasificación cargado en {device}.")
        except Exception as e:
            print(f"Error cargando el modelo de clasificación: {e}")

# =========
# Endpoints GAN  (sin cambios)
# =========
@app.post("/generateSingle")
async def generate_image(image: Image):
    seed = int(image.seed)
    truncation_psi = image.truncation_psi
    noise_mode = image.noise_mode
    G = get_generator(image.model)
    label = torch.zeros([1, G.c_dim], device=device)
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")
    return {"image": img_to_b64(pil_img), "seed": seed,
            "truncation_psi": truncation_psi, "noise_mode": noise_mode,
            "model": image.model}

@app.post("/generateSeveral")
async def generate_several(image: Image):
    truncation_psi = image.truncation_psi
    noise_mode = image.noise_mode
    G = get_generator(image.model)
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
            "truncation_psi": truncation_psi, "noise_mode": noise_mode,
            "model": image.model}

# =========
# /clasificar  — Petroglifo vs Pictograma (EfficientNet)
# =========
@app.post("/clasificar")
async def clasificar_imagen(imagen: UploadFile = File(...)):
    nombre = imagen.filename or ""
    if not nombre.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen (png, jpg, jpeg, bmp, webp).",
        )

    file_bytes = await imagen.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    try:
        img_rgb = read_image_rgb(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    clase, confianza, probabilidades = classify_image(PIL.Image.fromarray(img_rgb))

    return {
        "filename": nombre,
        "clase": clase,
        "confianza": round(confianza, 4),
        "confianza_porcentaje": round(confianza * 100, 2),
        "alta_confianza": confianza >= 0.75,
        "probabilidades": probabilidades,
        "clases": CLASS_NAMES,
    }

# =========
# /comparar  — ahora compara 256×256 vs 512×512 con el modelo ONNX
# =========
@app.post("/comparar")
async def comparar_modelos(
    imagen: UploadFile = File(...),
    model: OnnxModelName = Form("mejor_modelo_dinamico"),
):
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

    seg_threshold = (
        MASK_THRESHOLD if model == "mejor_modelo_dinamico" else GAB_MASK_THRESHOLD
    )

    # ── Inferencia con ambas resoluciones ────────────────────────────────
    mask_1, resultado_1, cob_1 = inferir_y_simular_onnx(
        img_rgb, img_size=256, threshold=seg_threshold, label="256×256",
        model_name=model)
    mask_2, resultado_2, cob_2 = inferir_y_simular_onnx(
        img_rgb, img_size=512, threshold=seg_threshold, label="512×512",
        model_name=model)

    # ── Figura comparación 2×3  (layout idéntico al original) ────────────
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(img_rgb);             axes[0, 0].set_title("Imagen original");                        axes[0, 0].axis("off")
    axes[0, 1].imshow(mask_1, cmap="gray"); axes[0, 1].set_title(f"Máscara — 256×256 ({cob_1:.1f}%)");     axes[0, 1].axis("off")
    axes[0, 2].imshow(resultado_1);         axes[0, 2].set_title("Simulación — 256×256");                   axes[0, 2].axis("off")

    axes[1, 0].imshow(img_rgb);             axes[1, 0].set_title("Imagen original");                        axes[1, 0].axis("off")
    axes[1, 1].imshow(mask_2, cmap="gray"); axes[1, 1].set_title(f"Máscara — 512×512 ({cob_2:.1f}%)");     axes[1, 1].axis("off")
    axes[1, 2].imshow(resultado_2);         axes[1, 2].set_title("Simulación — 512×512");                   axes[1, 2].axis("off")

    fig1.suptitle(f"Comparación de resoluciones — {model}",
                  fontsize=15, fontweight="bold")
    plt.tight_layout()
    img_comparacion = fig_to_base64(fig1)

    # ── Respuesta JSON  (mismas claves que el endpoint original) ─────────
    return JSONResponse(content={
        "model": model,
        "metricas": {
            "cobertura_modelo_1": round(cob_1, 4),   # resolución 256
            "cobertura_modelo_2": round(cob_2, 4),   # resolución 512
            "threshold_modelo_1": seg_threshold,
            "threshold_modelo_2": seg_threshold,
        },
        "imagenes": {
            "comparacion":         img_comparacion,
            "simulacion_modelo_1": pil_to_base64(resultado_1),
            "simulacion_modelo_2": pil_to_base64(resultado_2),
        }
    })