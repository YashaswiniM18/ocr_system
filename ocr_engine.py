import os
import hashlib


from paddleocr import PaddleOCR
import cv2
import numpy as np

# ── Configurable via environment variables ──────────────────────────────────
MAX_DIM = int(os.getenv("OCR_MAX_DIM", "1000"))   # lower = faster, less accurate on tiny text
MIN_DIM = int(os.getenv("OCR_MIN_DIM", "600"))    # below this we upscale

# ── Singleton OCR instance ──────────────────────────────────────────────────
_ocr_instance = None

def is_ocr_ready() -> bool:
    return _ocr_instance is not None

def get_ocr() -> PaddleOCR:
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            device="cpu",
            enable_mkldnn=True
        )
    return _ocr_instance

# ── In-process OCR result cache (keyed on fast structural hash) ────────────
_ocr_cache: dict = {}
_cache_hits: int = 0
_CACHE_MAX = 500

def get_cache_stats() -> dict:
    return {"size": len(_ocr_cache), "hits": _cache_hits}

def _fast_img_key(image: np.ndarray):
    """Fast cache key: shape + a few corner/center pixel samples.
    Far faster than md5(image.tobytes()) on large images."""
    h, w = image.shape[:2]
    try:
        samples = (
            image[0, 0].tobytes(),
            image[h // 2, w // 2].tobytes(),
            image[-1, -1].tobytes(),
            image[0, -1].tobytes(),
            image[-1, 0].tobytes(),
        )
        return (h, w, image.dtype.str, *samples)
    except Exception:
        return None

def run_ocr(image: np.ndarray):
    global _cache_hits

    if image is None:
        return [], []

    # ── Cache lookup ─────────────────────────────────────────────────────────
    img_key = _fast_img_key(image)
    if img_key is not None and img_key in _ocr_cache:
        _cache_hits += 1
        return _ocr_cache[img_key]

    # ── Resolution normalisation ─────────────────────────────────────────────
    h, w = image.shape[:2]
    upscale_factor = 1.0

    if h < MIN_DIM or w < MIN_DIM:
        upscale_factor = 1500.0 / h if h < w else 1500.0 / w
        if upscale_factor > 1.0:
            image = cv2.resize(image, None, fx=upscale_factor, fy=upscale_factor,
                               interpolation=cv2.INTER_CUBIC)
    elif h > MAX_DIM or w > MAX_DIM:
        downscale_factor = MAX_DIM / h if h > w else MAX_DIM / w
        if downscale_factor < 1.0:
            image = cv2.resize(image, None, fx=downscale_factor, fy=downscale_factor,
                               interpolation=cv2.INTER_AREA)
            upscale_factor = downscale_factor   # reuse to scale boxes back

    # ── Run OCR ──────────────────────────────────────────────────────────────
    ocr = get_ocr()
    result = ocr.ocr(image)

    if not result or result[0] is None:
        return [], []

    texts = []
    raw_data = []

    for line in result[0]:
        if len(line) < 2:
            continue

        box = line[0]
        text_info = line[1]

        if upscale_factor != 1.0:
            new_box = []
            if isinstance(box, (list, tuple)):
                for pt in box:
                    try:
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                            new_box.append([float(pt[0]) / upscale_factor,
                                            float(pt[1]) / upscale_factor])
                        else:
                            new_box.append([0.0, 0.0])
                    except (ValueError, TypeError, IndexError):
                        new_box.append([0.0, 0.0])
                box = new_box
            else:
                box = [[0, 0], [0, 0], [0, 0], [0, 0]]

        if not isinstance(text_info, (list, tuple)) or len(text_info) < 1:
            continue

        text = str(text_info[0])
        conf = 0.5
        if len(text_info) > 1:
            try:
                conf = float(text_info[1])
            except (ValueError, TypeError):
                pass

        texts.append(text)
        raw_data.append({"text": text, "box": box, "conf": conf})

    output = (texts, raw_data)

    # ── Store in cache ────────────────────────────────────────────────────────
    if img_key is not None:
        if len(_ocr_cache) >= _CACHE_MAX:
            # Evict oldest half when limit reached
            keys = list(_ocr_cache.keys())
            for k in keys[:len(keys) // 2]:
                del _ocr_cache[k]
        _ocr_cache[img_key] = output

    return output
