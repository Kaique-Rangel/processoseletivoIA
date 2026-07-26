import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 2 — Otimização do Modelo (CIFAR-10)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

MODEL_PATH = "model.h5"
TFLITE_PATH = "model.tflite"

# ---------------------------
# Carregar modelo treinado
# ---------------------------
print(f"Carregando modelo treinado de '{MODEL_PATH}'...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo carregado com sucesso.\n")

# ---------------------------
# Conversão para TensorFlow Lite
# ---------------------------
print("Convertendo modelo para TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
print("Conversão concluída.\n")

# ---------------------------
# Salvar modelo otimizado
# ---------------------------
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

# ---------------------------
# Relatório de resultados
# ---------------------------
h5_size_kb = os.path.getsize(MODEL_PATH) / 1024
tflite_size_kb = os.path.getsize(TFLITE_PATH) / 1024
reduction_pct = (1 - tflite_size_kb / h5_size_kb) * 100

print("Modelo convertido com sucesso!")
print(f"Arquivo gerado: {TFLITE_PATH}")
print("-" * 40)
print(f"Tamanho model.h5:     {h5_size_kb:10.2f} KB")
print(f"Tamanho model.tflite: {tflite_size_kb:10.2f} KB")
print(f"Redução de tamanho:   {reduction_pct:9.2f} %")
print("-" * 40)

# ---------------------------
# Sanity check: garante que o .tflite carrega e roda no Interpreter
# ---------------------------
print("\nValidando o modelo .tflite com tf.lite.Interpreter...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input:  shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
print(f"Output: shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")
print("Modelo .tflite validado com sucesso — pronto para uso em run_inference.py.")