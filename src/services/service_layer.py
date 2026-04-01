import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from src.services.explainable_ai_service import ExplainableAIService

BACKBONE_LAYER_NAME = 'efficientnetv2-s'
explainable_ai_service = ExplainableAIService()


class ServiceLayer:
    def __init__(self):
        self.model_path="C:\\Users\\User\\Desktop\\Dev\\doctor_dr_model_service\\src\\models\\efficientnetV2_S_fined_tunned_phase_model_recent_experiment_2_v1.keras"
        self.model = load_model(self.model_path)
        self.img_size = (384, 384)

    def predict(self,image):
        preprocess_image =self.preprocess_image(image)
        print(self.model)
        backbone=self.model.get_layer(BACKBONE_LAYER_NAME)

        explainable_ai_service.get_last_layers_of_backbone(backbone)

        result=explainable_ai_service.generate_heat_map( self.model,preprocess_image )
        overlay = explainable_ai_service.overlay_gradcam(image, result["heatmap"])

        # explainable_ai_service.plot_gradcam_result(overlay)
        # prediction = self.model.predict(preprocess_image)
        # print(prediction)

        return {
            "overlay": overlay,
            "image": image,
            "pred_index": result["pred_index"],
            "pred_probs": result["pred_probs"],
        }


    def dummy_predict(self):
        image=Image.open("E:\\reformated_dataset\\train\\0_class_007-0008-000.jpg")
        return self.predict(image)

    def preprocess_image(self, image):
        # Ensure RGB
        image = image.convert("RGB")

        # Resize to model input size
        image = image.resize(self.img_size)

        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)

        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        image_array = np.expand_dims(image_array, axis=0)

        return image_array