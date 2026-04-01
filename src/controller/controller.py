from flask import Flask, request, jsonify
from src.services.service_layer import ServiceLayer
from src.controller.controller_helper import prepare_gradcam_for_json

#
app = Flask(__name__)

service_layer_instance = ServiceLayer()


@app.route('/dr-predict',methods=['POST'])
def dr_predict():
    # Use request.get_json() to parse the incoming JSON data
    data = request.get_json()

    # Check if the data is a valid JSON object
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    print(data)
    return  "HI"

@app.route('/get',methods=['GET'])
def get():
    result =service_layer_instance.dummy_predict()
    overlay = result["overlay"]
    print(type(overlay))
    overlay_base64 = prepare_gradcam_for_json(overlay)
    class_probabilities =result["pred_probs"]

    return jsonify({
        "data":{
            "prediction_index": int(result["pred_index"]),
            "class_probability":{
              "class_01": float(class_probabilities[0]),
              "class_02": float(class_probabilities[1]),
              "class_03": float(class_probabilities[2]),
              "class_04": float(class_probabilities[3]),
              "class_05": float(class_probabilities[4]),
              "class_06": float(class_probabilities[5]),
            },
            "overlay_image": overlay_base64,
        },
        "message": "successful prediction"
    })



if __name__ == '__main__':
    app.run()


