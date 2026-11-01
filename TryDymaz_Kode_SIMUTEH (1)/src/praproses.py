"""Preprocessing pipeline for SIMUTEH, version v0.1-praproses."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

PARAM_BAWAAN = {
    "lebar": 640,
    "gamma": 1.0,
    "clahe_clip": 0.0,
    "clahe_tile": 8,
    "penapis": "median",
    "kernel": 3,
    "sigma": 1.0,
    "tajam_k": 0.0,
}

DEFAULT_CFG = {
    "kanal_evaluasi": "R",
    "dasar": {**PARAM_BAWAAN},
    "per_kondisi": {
        "terang": {"gamma": 1.6},
        "bayangan": {"clahe_clip": 1.0, "clahe_tile": 8},
    },
    "ambang": {
        "bayangan_lr": 40,
        "terang_rerata": 190,
        "kontras_std": 25,
        "gelap_rerata": 145,
    },
}


def _resize(img_bgr: np.ndarray, lebar: int) -> np.ndarray:
    if img_bgr is None or not isinstance(img_bgr, np.ndarray):
        raise ValueError("img_bgr harus berupa array citra yang valid")
    if img_bgr.ndim not in (2, 3):
        raise ValueError("Citra harus 2D (grayscale) atau 3D (BGR)")
    h, w = img_bgr.shape[:2]
    if lebar <= 0:
        raise ValueError("Parameter lebar harus lebih besar dari nol")
    tinggi = max(1, int(round(h * lebar / w)))
    if w == lebar:
        return img_bgr.copy()
    return cv2.resize(img_bgr, (lebar, tinggi), interpolation=cv2.INTER_AREA)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("Gamma harus lebih besar dari nol")
    lut = np.array([255.0 * (i / 255.0) ** gamma for i in range(256)], dtype=np.float32)
    return cv2.LUT(img, np.clip(np.round(lut), 0, 255).astype(np.uint8))


def _clahe_v(img_bgr: np.ndarray, clip: float, tile: int) -> np.ndarray:
    if clip <= 0:
        return img_bgr
    if tile < 1:
        raise ValueError("clahe_tile minimal 1")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _denoise(img: np.ndarray, method: str, kernel: int, sigma: float) -> np.ndarray:
    method = str(method).lower()
    if method in ("none", "tanpa", "off"):
        return img
    if kernel < 1:
        raise ValueError("Ukuran kernel minimal 1")
    if method == "median":
        if kernel % 2 == 0:
            raise ValueError("Kernel median harus ganjil")
        return cv2.medianBlur(img, int(kernel))
    if method == "rerata":
        return cv2.blur(img, (int(kernel), int(kernel)))
    if method == "gaussian":
        if kernel % 2 == 0:
            raise ValueError("Kernel Gaussian harus ganjil")
        return cv2.GaussianBlur(img, (int(kernel), int(kernel)), float(sigma))
    raise ValueError(f"Metode penapis tidak dikenal: {method}")


def _unsharp(img: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return img
    blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    return cv2.addWeighted(img, 1.0 + float(strength), blur, -float(strength), 0)


def praproses(img_bgr: np.ndarray, p: dict | None = None) -> np.ndarray:
    """Resize -> gamma -> CLAHE V -> denoise -> optional unsharp masking."""
    p = {**PARAM_BAWAAN, **(p or {})}
    img = _resize(img_bgr, int(p["lebar"]))
    if float(p["gamma"]) != 1.0:
        img = _gamma(img, float(p["gamma"]))
    if float(p["clahe_clip"]) > 0:
        if img.ndim == 2:
            img = cv2.createCLAHE(
                clipLimit=float(p["clahe_clip"]),
                tileGridSize=(int(p["clahe_tile"]), int(p["clahe_tile"])),
            ).apply(img)
        else:
            img = _clahe_v(img, float(p["clahe_clip"]), int(p["clahe_tile"]))
    img = _denoise(img, str(p["penapis"]), int(p["kernel"]), float(p["sigma"]))
    img = _unsharp(img, float(p["tajam_k"]))
    return img


def fitur_cahaya(img: np.ndarray) -> tuple[float, float, float]:
    gray = _to_gray(img).astype(np.float32)
    w = gray.shape[1] // 4
    if w == 0:
        lr = 0.0
    else:
        lr = abs(float(gray[:, :w].mean()) - float(gray[:, -w:].mean()))
    return float(gray.mean()), float(gray.std()), lr


def deteksi_kondisi(img: np.ndarray, ambang: dict | None = None) -> str:
    """Classify synthetic lighting using mean, standard deviation and left-right shift."""
    a = {**DEFAULT_CFG["ambang"], **(ambang or {})}
    mean, std, lr = fitur_cahaya(img)
    if lr > float(a["bayangan_lr"]):
        return "bayangan"
    if mean > float(a["terang_rerata"]):
        return "terang"
    if std < float(a["kontras_std"]):
        return "kontras_rendah"
    if mean < float(a["gelap_rerata"]):
        return "gelap"
    return "normal"


def praproses_adaptif(img_bgr: np.ndarray, cfg: dict | None = None) -> tuple[np.ndarray, str]:
    cfg = cfg or DEFAULT_CFG
    dasar = {**PARAM_BAWAAN, **cfg.get("dasar", {})}
    # Detect after resizing to the actual working resolution.
    small = _resize(img_bgr, int(dasar["lebar"]))
    kondisi = deteksi_kondisi(small, cfg.get("ambang"))
    params = {**dasar, **cfg.get("per_kondisi", {}).get(kondisi, {})}
    return praproses(img_bgr, params), kondisi


def simpan_param(p: dict, path: str = "../hasil/param_praproses.json") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({**PARAM_BAWAAN, **p}, indent=2), encoding="utf-8")


def simpan_cfg(cfg: dict, path: str = "../hasil/param_praproses.json") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def muat_param(path: str = "../hasil/param_praproses.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
