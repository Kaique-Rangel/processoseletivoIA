import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ---------------------------------------------------------------------------
# Projeto 2 — Classificação CIFAR-10
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset CIFAR-10 via tf.keras.datasets.cifar10
#   2. Normalizar as imagens para [0, 1] (shape (32, 32, 3))
#   3. Separar um conjunto de validação
#   4. Incluir data augmentation (ex: layers.RandomFlip, RandomRotation, RandomZoom)
#      aplicada ao conjunto de treino
#   5. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   6. Treinar com EarlyStopping monitorando a perda de validação
#   7. Exibir a acurácia de validação final no terminal
#   8. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

tf.keras.utils.set_random_seed(42)

# ---------------------------
# Carregamento do dataset
# ---------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# ---------------------------
# Normalização das imagens
# ---------------------------
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = y_train.squeeze()
y_test = y_test.squeeze()

# ---------------------------
# Separação do conjunto de validação
# ---------------------------
VALIDATION_SIZE = 5000

x_val = x_train[-VALIDATION_SIZE:]
y_val = y_train[-VALIDATION_SIZE:]

x_train = x_train[:-VALIDATION_SIZE]
y_train = y_train[:-VALIDATION_SIZE]

# ---------------------------
# Data Augmentation
# ---------------------------
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)

L2 = regularizers.l2(1e-4)


def conv_block(x, filters, dropout_rate):
   
    x = layers.Conv2D(filters, (3, 3), padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=L2, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=L2, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

# ---------------------------
# Construção do modelo CNN
# ---------------------------
inputs = keras.Input(shape=(32, 32, 3))
x = data_augmentation(inputs)

x = conv_block(x, 32, 0.2)
x = conv_block(x, 64, 0.3)
x = conv_block(x, 128, 0.4)
x = conv_block(x, 256, 0.4)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    kernel_regularizer=L2,
    kernel_initializer="he_normal",
    use_bias=False,
)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(
    10,
    activation="softmax",
    kernel_initializer="glorot_uniform",
)(x)

model = keras.Model(inputs, outputs)
model.summary(line_length=120)

# ---------------------------
# Compilação do modelo
# ---------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ---------------------------
# Callbacks
# ---------------------------
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1,
)

checkpoint = keras.callbacks.ModelCheckpoint(
    "model.h5",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1,
)

# ---------------------------
# Treinamento
# ---------------------------
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    verbose=1,
)

# ---------------------------
# Avaliação
# ---------------------------
val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
print(f"\nAcurácia final de validação: {val_accuracy:.4f}")

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia final de teste: {test_accuracy:.4f}")

print("\nTreinamento finalizado.")
print("Melhor modelo salvo como model.h5")