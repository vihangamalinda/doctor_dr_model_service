import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from  src.services.image_file_convertor import ImageFileConvertor

class ExplainableAIService:
    def __init__(self):
        self.img_size =384
        self.backbone_name ='efficientnetv2-s'
        self.last_conv_layer_name = "top_conv"

    @staticmethod
    def get_last_layers_of_backbone(backbone):
        for layer in backbone.layers[-30:]:
            print(layer.name, layer.output.shape)


    def overlay_gradcam(self,image, heatmap, alpha=0.4):
        image =ImageFileConvertor.convert_jpeg_to_numpy(image)

        size = (self.img_size, self.img_size)
        # image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # ✅ Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")

        # Convert BGR → RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, size)

        # Resize heatmap
        heatmap = cv2.resize(heatmap, size)
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

        return superimposed_img

    def make_gradcam_heatmap(self, img_array, full_model, pred_index=None):
        import tensorflow as tf
        import numpy as np

        backbone = full_model.get_layer(self.backbone_name)
        last_conv_layer = backbone.get_layer(self.last_conv_layer_name)

        # Model: backbone input -> chosen conv layer output
        last_conv_layer_model = tf.keras.Model(
            inputs=backbone.input,
            outputs=last_conv_layer.output
        )

        # Model: chosen conv layer output -> backbone final output
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input

        found = False
        for layer in backbone.layers:
            if layer.name == self.last_conv_layer_name:
                found = True
                continue
            if found:
                x = layer(x)

        backbone_tail_model = tf.keras.Model(classifier_input, x)

        # Model: backbone output -> full model final prediction
        classifier_output_input = tf.keras.Input(shape=backbone.output.shape[1:])
        y = classifier_output_input

        passed_backbone = False
        for layer in full_model.layers:
            if layer.name == self.backbone_name:
                passed_backbone = True
                continue
            if passed_backbone:
                y = layer(y)

        top_model = tf.keras.Model(classifier_output_input, y)

        with tf.GradientTape() as tape:
            # Forward pass to chosen conv layer
            last_conv_output = last_conv_layer_model(img_array, training=False)
            tape.watch(last_conv_output)

            # Forward pass from chosen conv layer to backbone output
            backbone_output = backbone_tail_model(last_conv_output, training=False)

            # Final prediction from classification head
            preds = top_model(backbone_output, training=False)

            # Save full probability array
            pred_probs = preds[0].numpy()

            # Use predicted class if not given
            if pred_index is None:
                pred_index = int(tf.argmax(preds[0]).numpy())
            else:
                pred_index = int(pred_index)

            # Score for the target class
            class_channel = preds[:, pred_index]

        # Gradients of target class wrt chosen conv layer
        grads = tape.gradient(class_channel, last_conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_output = last_conv_output[0]
        heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Keep only positive influence
        heatmap = tf.maximum(heatmap, 0)

        max_val = tf.reduce_max(heatmap)
        if max_val == 0:
            return {
                "heatmap": np.zeros(heatmap.shape, dtype=np.float32),
                "pred_index": pred_index,
                "pred_probs": pred_probs
            }

        heatmap = heatmap / max_val

        return {
            "heatmap": heatmap.numpy(),
            "pred_index": pred_index,
            "pred_probs": pred_probs
        }

    def load_and_prepare_image(self,img_path):
        img_size =self.img_size
        img = tf.keras.utils.load_img(img_path, target_size=(img_size, img_size))
        img = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img, axis=0)

        # same preprocessing used in model
        # img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        return img_array


    def generate_heat_map(self, model,img_array):
        # img_array = self.load_and_prepare_image("test_image.jpg")


        result = self.make_gradcam_heatmap(
            img_array=img_array,
            full_model=model,
            pred_index=None
        )
        return result

    def plot_gradcam_result(self, overlay_image, title="Grad-CAM Result"):
        """
        overlay_image: output from overlay_gradcam (numpy array, RGB)
        """

        if overlay_image is None:
            raise ValueError("overlay_image is None")

        if not isinstance(overlay_image, np.ndarray):
            raise ValueError("overlay_image must be a numpy array")

        plt.figure(figsize=(6, 6))
        plt.imshow(overlay_image)
        plt.title(title)
        plt.axis("off")
        plt.show()
