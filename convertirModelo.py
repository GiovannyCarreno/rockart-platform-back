import tensorflow as tf
import tf2onnx
import onnx
from tensorflow.keras import backend as K

# =========================
# DEFINIR FUNCIONES PERSONALIZADAS
# =========================

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return 1 - (
        (2. * intersection + smooth) /
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    )

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss

# =========================
# CARGAR MODELO
# =========================

model = tf.keras.models.load_model(
    "mejor_modelo.keras",
    custom_objects={
        "bce_dice_loss": bce_dice_loss
    },
    compile=False
)

# =========================
# ENTRADA DINÁMICA
# =========================

input_signature = [
    tf.TensorSpec(
        shape=(None, None, None, 3),
        dtype=tf.float32,
        name="input"
    )
]

# =========================
# CONVERTIR A ONNX
# =========================

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13
)

# =========================
# GUARDAR MODELO
# =========================

output_path = "modelo_dinamico_gab.onnx"

with open(output_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Modelo ONNX guardado en: {output_path}")

# =========================
# VERIFICAR
# =========================

model_onnx = onnx.load(output_path)

print("\nEntradas del modelo ONNX:")

for inp in model_onnx.graph.input:
    print(inp)