import numpy as np
import cv2
import base64
from PIL import Image
import tensorflow as tf

class ImageFileConvertor:
    def __init__(self):
        pass

    @staticmethod
    def convert_jpeg_to_numpy(image):

        # 1. Already a numpy image
        if isinstance(image, np.ndarray):
            return image

        # 2. Flask FileStorage (request.files['image'])
        if hasattr(image, "read"):
            image.stream.seek(0)  # important: reset pointer
            file_bytes = np.frombuffer(image.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from FileStorage")
            return img

        # 3. Raw bytes
        if isinstance(image, (bytes, bytearray)):
            file_bytes = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from bytes")
            return img

        # 4. Base64 string (common in APIs)
        if isinstance(image, str):
            try:
                # Remove header if exists (data:image/jpeg;base64,...)
                if "," in image:
                    image = image.split(",")[1]

                decoded = base64.b64decode(image)
                file_bytes = np.frombuffer(decoded, np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Failed to decode base64 image")
                return img
            except Exception:
                raise ValueError("Invalid base64 image string")

        # 5. PIL Image
        try:

            if isinstance(image, Image.Image):
                img = np.array(image)
                # Convert RGB → BGR (OpenCV format)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img
        except ImportError:
            pass

        # 6. TensorFlow tensor
        try:

            if isinstance(image, tf.Tensor):
                img = image.numpy()

                # remove batch if exists
                if len(img.shape) == 4:
                    img = img[0]

                # convert float → uint8 if needed
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)

                return img
        except ImportError:
            pass

        raise ValueError(f"Unsupported image type: {type(image)}")