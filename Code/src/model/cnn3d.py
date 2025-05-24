import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def load_data(X_path="outputs/X.npy", y_path="outputs/y.npy"):
    X = np.load(X_path)
    if X.ndim == 4:
        X = X[..., np.newaxis]  # Asegura (samples, H, W, D, 1)
    y = np.load(y_path)
    return X, y

def build_cnn3d_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv3D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv3D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_cnn3d_model(X, y, output_dir="outputs", model_name="cnn3d_model", epochs=30, batch_size=32):
    """
    Entrena un modelo CNN 3D (binario o multiclase) y lo guarda con el accuracy en el nombre.

    Parámetros:
        X: numpy array de volúmenes (N, H, W, D, 1)
        y: etiquetas (one-hot o binario)
        output_dir: carpeta donde se guardarán los archivos
        model_name: nombre base del modelo
        epochs: número de épocas
        batch_size: tamaño de lote
    """

    # Detectar si y es one-hot (2D) o etiquetas simples (1D)
    if y.ndim == 2:
        stratify_values = y.argmax(axis=1)
    else:
        stratify_values = y

    # 1. Dividir en 70% train y 30% temporal
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, stratify=stratify_values, test_size=0.30, random_state=42
    )

    # 2. Dividir temporal en 10% val y 20% test
    if y_temp.ndim == 2:
        stratify_temp = y_temp.argmax(axis=1)
    else:
        stratify_temp = y_temp

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, stratify=stratify_temp, test_size=2/3, random_state=42
    )

    # Construir el modelo
    from src.model.cnn3d import build_cnn3d_model  # asegúrate de que está disponible
    model = build_cnn3d_model(input_shape=X.shape[1:], num_classes=y.shape[1] if y.ndim == 2 else 1)

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Entrenar
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # Guardar historial
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "history.pkl"), "wb") as f:
        pickle.dump(history.history, f)

    # Evaluar
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test accuracy: {acc:.4f}")

    # Guardar modelo con accuracy en el nombre
    acc_percent = round(acc * 100)
    model_filename = f"{model_name}_{acc_percent}.keras"
    model_path = os.path.join(output_dir, model_filename)
    model.save(model_path)
    print(f"💾 Modelo guardado en: {model_path}")

    return model