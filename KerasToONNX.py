from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
import onnx
import keras2onnx

base_model_path = "model.hdf5"
onnx_model_name = 'model.onnx'

model = load_model(base_model_path)
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, onnx_model_name)
