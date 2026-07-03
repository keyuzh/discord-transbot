import os
import base64

import requests

# =========================
# CONFIG
# =========================

_BACKEND = os.getenv("OCR_BACKEND", "paddleocr").lower()

# --- paddleocr backend (local, CPU) ---
# PP-OCRv5's default ("ch") recognition model already covers Simplified
# Chinese, Traditional Chinese, English, and Japanese in one pipeline.
# Korean and Spanish fall outside that set and need their own pipelines.
_PIPELINE_LANGS = {
    "default": "ch",
    "korean": "korean",
    "es": "es",
}
_CONFIDENCE_THRESHOLD = 0.6
_pipelines = {}

# --- ocr.space backend (cloud) ---
_OCR_SPACE_URL = "https://api.ocr.space/parse/image"
_OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")
_OCR_SPACE_LANGUAGE = os.getenv("OCR_SPACE_LANGUAGE", "auto")

# --- google cloud vision backend (cloud) ---
_GCV_URL = "https://vision.googleapis.com/v1/images:annotate"
_GCV_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")


def init_ocr():
    """Prepare whichever backend OCR_BACKEND selects. Blocking; call once at startup."""
    if _BACKEND == "paddleocr":
        _init_paddleocr()
    elif _BACKEND == "ocrspace":
        if not _OCR_SPACE_API_KEY:
            raise RuntimeError("OCR_BACKEND=ocrspace requires OCR_SPACE_API_KEY to be set.")
    elif _BACKEND == "gcv":
        if not _GCV_API_KEY:
            raise RuntimeError("OCR_BACKEND=gcv requires GOOGLE_VISION_API_KEY to be set.")
    else:
        raise ValueError(f"Unknown OCR_BACKEND '{_BACKEND}'. Expected 'paddleocr', 'ocrspace', or 'gcv'.")


def extract_image_text(image_bytes, content_type=None):
    """Run OCR against image bytes using the configured backend. Blocking."""
    if _BACKEND == "paddleocr":
        return _extract_paddleocr(image_bytes)
    elif _BACKEND == "ocrspace":
        return _extract_ocrspace(image_bytes, content_type or "image/png")
    elif _BACKEND == "gcv":
        return _extract_gcv(image_bytes)
    return ""


# =========================
# PADDLEOCR (local, CPU)
# =========================

def _init_paddleocr():
    from paddleocr import PaddleOCR

    for name, lang in _PIPELINE_LANGS.items():
        _pipelines[name] = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # Works around a paddlepaddle 3.3.x oneDNN/PIR regression that
            # raises NotImplementedError during CPU inference otherwise.
            enable_mkldnn=False,
        )


def _predict(pipeline, image_bytes):
    """Run one pipeline against image bytes. Returns (text, avg_confidence)."""
    import numpy as np
    import cv2

    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return "", 0.0

    results = pipeline.predict(img)

    texts = []
    scores = []
    for res in results:
        for rec_text, rec_score in zip(res.get("rec_texts", []), res.get("rec_scores", [])):
            # skip blank recognitions (e.g. a script the model can't read
            # yields empty/whitespace text but a misleadingly non-zero score)
            if rec_text.strip():
                texts.append(rec_text)
                scores.append(rec_score)

    if not texts:
        return "", 0.0

    text = "\n".join(texts)
    avg_confidence = sum(scores) / len(scores)
    return text, avg_confidence


def _extract_paddleocr(image_bytes):
    """
    Run OCR against image bytes, trying the default (en/ja/zh-cn/zh-tw)
    pipeline first and only falling through to Korean/Spanish pipelines
    when the default result is empty or low-confidence.
    """
    best_text, best_confidence = "", 0.0

    for name in ("default", "korean", "es"):
        pipeline = _pipelines.get(name)
        if pipeline is None:
            continue

        text, confidence = _predict(pipeline, image_bytes)

        if confidence > best_confidence:
            best_text, best_confidence = text, confidence

        if best_text and best_confidence >= _CONFIDENCE_THRESHOLD:
            break

    return best_text


# =========================
# OCR.SPACE (cloud)
# =========================

def _extract_ocrspace(image_bytes, content_type):
    try:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        resp = requests.post(
            _OCR_SPACE_URL,
            data={
                "apikey": _OCR_SPACE_API_KEY,
                "base64Image": f"data:{content_type};base64,{b64}",
                "language": _OCR_SPACE_LANGUAGE,
                "OCREngine": 2,
                "scale": True,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("IsErroredOnProcessing"):
            print(f"OCR.space error: {data.get('ErrorMessage')}")
            return ""

        texts = [
            r.get("ParsedText", "")
            for r in data.get("ParsedResults") or []
            if r.get("ParsedText", "").strip()
        ]
        return "\n".join(texts).strip()
    except requests.RequestException as e:
        print(f"OCR.space request failed: {e}")
        return ""


# =========================
# GOOGLE CLOUD VISION (cloud)
# =========================

def _extract_gcv(image_bytes):
    try:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        resp = requests.post(
            _GCV_URL,
            params={"key": _GCV_API_KEY},
            json={
                "requests": [
                    {
                        "image": {"content": b64},
                        "features": [{"type": "TEXT_DETECTION"}],
                    }
                ]
            },
            timeout=30,
        )
        resp.raise_for_status()
        result = (resp.json().get("responses") or [{}])[0]

        if "error" in result:
            print(f"Google Vision error: {result['error']}")
            return ""

        return result.get("fullTextAnnotation", {}).get("text", "").strip()
    except requests.RequestException as e:
        print(f"Google Vision request failed: {e}")
        return ""
