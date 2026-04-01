import io
import base64
from PIL import Image
import numpy as np

def prepare_gradcam_for_json(overlay: np.ndarray) -> str:

    if overlay.ndim == 2:
        overlay = np.stack([overlay]*3, axis=-1)

    if overlay.dtype != np.uint8:
        overlay = (overlay * 255).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(overlay)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")
