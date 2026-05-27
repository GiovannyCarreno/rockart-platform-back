"""Postprocesado de máscaras de segmentación de petroglifos."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_AREA = 150
DEFAULT_LINE_WIDTH = 2
DEFAULT_THRESHOLD_CANDIDATES = (0.4, 0.5, 0.6)

PROJECT_ROOT = Path(__file__).resolve().parent
BACKGROUND_DIR = PROJECT_ROOT / "fondo"


def preprocess(img_bgr: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def limpiar_mascara(mask_bin: np.ndarray, min_area: int = DEFAULT_MIN_AREA) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    clean = np.zeros_like(mask_bin)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def fill_mask_holes(mask_bin: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = mask_bin.copy()
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def refine_mask(mask_bin: np.ndarray) -> np.ndarray:
    if not np.any(mask_bin):
        return mask_bin

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    refined = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    refined = fill_mask_holes(refined)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, open_kernel, iterations=1)
    refined = cv2.GaussianBlur(refined, (0, 0), sigmaX=1.2, sigmaY=1.2)
    refined = ((refined > 110).astype(np.uint8) * 255)
    return refined


def evaluate_mask(mask_bin: np.ndarray) -> dict[str, float | int | bool | list[str]]:
    h, w = mask_bin.shape[:2]
    total_pixels = h * w
    area_pixels = int((mask_bin > 0).sum())
    area_percent = area_pixels / total_pixels * 100.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_bin > 0).astype(np.uint8))
    component_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    component_count = len(component_areas)
    largest_component = max(component_areas, default=0)
    fragmented_ratio = 0.0 if area_pixels == 0 else 1.0 - (largest_component / max(area_pixels, 1))

    coords = cv2.findNonZero(mask_bin)
    touches_border = False
    bbox_fill = 0.0
    if coords is not None:
        x, y, bw, bh = cv2.boundingRect(coords)
        touches_border = x <= 2 or y <= 2 or (x + bw) >= w - 2 or (y + bh) >= h - 2
        bbox_fill = area_pixels / max(bw * bh, 1)

    score = 0.0
    warnings: list[str] = []

    if area_percent < 1.0:
        warnings.append("mask_too_small")
        score -= 8.0
    elif area_percent < 4.0:
        warnings.append("mask_small")
        score -= 2.5
    elif area_percent <= 38.0:
        score += 3.0
    elif area_percent <= 55.0:
        warnings.append("mask_large")
        score -= 1.2
    else:
        warnings.append("mask_too_large")
        score -= 6.0

    if component_count == 0:
        warnings.append("mask_empty")
        score -= 10.0
    elif component_count <= 3:
        score += 2.0
    elif component_count <= 6:
        score += 0.5
    else:
        warnings.append("fragmented_mask")
        score -= min(5.0, component_count * 0.4)

    if fragmented_ratio > 0.45:
        warnings.append("weak_main_component")
        score -= 3.0
    else:
        score += 1.8

    if bbox_fill < 0.08 and area_pixels > 0:
        warnings.append("sparse_mask")
        score -= 2.0
    elif bbox_fill > 0.18:
        score += 1.0

    if touches_border and area_percent > 20.0:
        warnings.append("touches_border")
        score -= 1.8

    is_valid = score >= 0.0 and "mask_empty" not in warnings and "mask_too_large" not in warnings

    return {
        "score": float(score),
        "area_percent": float(area_percent),
        "component_count": int(component_count),
        "largest_component": int(largest_component),
        "fragmented_ratio": float(fragmented_ratio),
        "bbox_fill": float(bbox_fill),
        "touches_border": bool(touches_border),
        "is_valid": bool(is_valid),
        "warnings": warnings,
    }


def select_best_mask(probability: np.ndarray, threshold: float, min_area: int) -> dict[str, object]:
    candidate_thresholds = []
    for t in (threshold, *DEFAULT_THRESHOLD_CANDIDATES):
        rounded = round(float(t), 3)
        if rounded not in candidate_thresholds:
            candidate_thresholds.append(rounded)

    candidates: list[dict[str, object]] = []
    for current_threshold in candidate_thresholds:
        base_mask = limpiar_mascara(
            (probability > current_threshold).astype(np.uint8) * 255,
            min_area=min_area,
        )
        refined_mask = refine_mask(base_mask)

        for strategy_name, candidate_mask in (
            ("raw", base_mask),
            ("refined", refined_mask),
        ):
            metrics = evaluate_mask(candidate_mask)
            candidates.append(
                {
                    "mask": candidate_mask,
                    "threshold": float(current_threshold),
                    "strategy": strategy_name,
                    "metrics": metrics,
                }
            )

    return max(candidates, key=lambda item: item["metrics"]["score"])  # type: ignore[index]


def adelgazar_mascara(mask_bin: np.ndarray, grosor: int = DEFAULT_LINE_WIDTH) -> np.ndarray:
    skeleton = skeletonize(mask_bin > 0).astype(np.uint8) * 255
    kernel_size = max(1, grosor)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(skeleton, kernel, iterations=max(1, grosor))


def build_relief_map(mask: np.ndarray) -> np.ndarray:
    mask_alpha = mask.astype(np.float32) / 255.0
    soft = cv2.GaussianBlur(mask_alpha, (0, 0), sigmaX=3.4, sigmaY=3.4)
    grad_x = cv2.Sobel(soft, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(soft, cv2.CV_32F, 0, 1, ksize=3)

    normal = np.dstack((-grad_x, -grad_y, np.ones_like(soft) * 0.7))
    normal /= np.linalg.norm(normal, axis=2, keepdims=True) + 1e-8

    light_dir = np.array([-0.45, -0.55, 0.7], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    shading = np.clip(np.sum(normal * light_dir, axis=2), 0.0, 1.0)

    center_dark = cv2.GaussianBlur(mask_alpha, (0, 0), sigmaX=5.5, sigmaY=5.5)
    edge_band = np.clip(soft - center_dark * 0.72, 0.0, 1.0)
    relief = (shading * 0.48) + (edge_band * 0.24) - (center_dark * 0.92)
    relief *= np.clip(soft * 1.4, 0.0, 1.0)
    return relief


def create_procedural_background(size: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    base = np.full((size, size, 3), (150, 134, 110), dtype=np.float32)

    noise_small = rng.normal(0, 1, (size // 4, size // 4, 3)).astype(np.float32)
    noise = cv2.resize(noise_small, (size, size), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), 4.5)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    radial = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size * 0.78)
    radial = np.clip(radial, 0.0, 1.0)[..., None]

    background = base + noise * 11.0
    background *= 1.0 - radial * 0.18

    warm_tint = np.array([1.02, 0.99, 0.94], dtype=np.float32)
    background *= warm_tint
    return np.clip(background, 0, 255).astype(np.uint8)


def estimate_background_style(background_rgb: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray | float]:
    bg = background_rgb.astype(np.float32)
    outside_mask = (mask == 0)

    if not np.any(outside_mask):
        outside_mask = np.ones(mask.shape, dtype=bool)

    pixels = bg[outside_mask]
    mean_color = pixels.mean(axis=0)
    std_color = pixels.std(axis=0)

    gray = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    texture_strength = float(np.mean(np.abs(laplacian)))

    return {
        "mean_color": mean_color,
        "std_color": std_color,
        "texture_strength": texture_strength,
    }


def choose_background(background_path: Path | None, size: int) -> tuple[np.ndarray, str]:
    if background_path is not None:
        bg = cv2.imread(str(background_path))
        if bg is None:
            raise FileNotFoundError(f"No se pudo leer el fondo: {background_path}")
        return (
            cv2.cvtColor(
                cv2.resize(bg, (size, size), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            ),
            background_path.name,
        )

    if BACKGROUND_DIR.exists():
        candidates = [
            p for p in BACKGROUND_DIR.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        if candidates:
            chosen_path = random.choice(candidates)
            bg = cv2.imread(str(chosen_path))
            if bg is not None:
                return (
                    cv2.cvtColor(
                        cv2.resize(bg, (size, size), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB,
                    ),
                    chosen_path.name,
                )

    return create_procedural_background(size), "procedural"


def render_petroglyph(background_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bg = background_rgb.astype(np.float32)
    relief = build_relief_map(mask)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=2.4, sigmaY=2.4)
    alpha = np.clip(alpha * 1.08, 0.0, 1.0)[..., None]
    style = estimate_background_style(background_rgb, mask)
    mean_color = style["mean_color"]
    std_color = style["std_color"]
    texture_strength = float(style["texture_strength"])

    warmth_shift = np.array([
        std_color[0] * 0.10,
        std_color[1] * 0.06,
        -std_color[2] * 0.08,
    ], dtype=np.float32)
    carved_tint = np.clip(mean_color + warmth_shift - 18.0, 0.0, 255.0)
    highlight_tint = np.clip(mean_color + np.array([16.0, 13.0, 10.0], dtype=np.float32), 0.0, 255.0)
    depth_strength = np.clip(0.20 + texture_strength / 140.0, 0.18, 0.36)
    relief_strength = np.clip(84.0 + texture_strength * 1.15, 88.0, 138.0)

    carved = bg.copy()
    carved = carved * (1.0 - alpha * depth_strength) + carved_tint[None, None, :] * (alpha * 0.10)
    carved += relief[..., None] * relief_strength

    inner_shadow = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), sigmaX=8.0, sigmaY=8.0)
    carved -= inner_shadow[..., None] * (16.0 + texture_strength * 0.22) * alpha

    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    edge_alpha = cv2.GaussianBlur(edge.astype(np.float32) / 255.0, (0, 0), sigmaX=2.0, sigmaY=2.0)[..., None]

    carved = carved * (1.0 - edge_alpha * 0.10) + highlight_tint[None, None, :] * (edge_alpha * 0.10)
    carved += edge_alpha * np.array([8.0, 6.5, 5.0], dtype=np.float32)

    gray = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    local_texture = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.8, sigmaY=1.8) - cv2.GaussianBlur(
        gray, (0, 0), sigmaX=6.0, sigmaY=6.0
    )
    carved += local_texture[..., None] * alpha * (18.0 + texture_strength * 0.10)

    return np.clip(carved, 0, 255).astype(np.uint8)
