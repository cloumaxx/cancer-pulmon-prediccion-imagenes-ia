import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_data(X_path="outputs/X.npy", y_path="outputs/y.npy"):
    X = np.load(X_path)
    X = X[..., np.newaxis]  # (samples, H, W, D, 1)
    y = np.load(y_path)
    y = to_categorical(y)  # one-hot
    return X, y

def build_cnn3d(input_shape, num_classes):
    model = Sequential([
        Conv3D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling3D(pool_size=2),
        Dropout(0.2),
        Conv3D(64, kernel_size=3, activation='relu'),
        MaxPooling3D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax') # 4 clases: No cáncer, Adeno, Escamoso, Otro
    ])

    """
    Capa	            Función
    Conv3D(32)	        Detecta patrones espaciales 3D locales (texturas, formas) usando 32 filtros 3D.
    MaxPooling3D(2) 	Reduce el tamaño espacial del volumen, conservando las características más destacadas.
    Dropout(0.2)	    Apaga aleatoriamente el 20% de las neuronas para evitar overfitting.
    Conv3D(64)	        Aprende patrones más abstractos y complejos con más filtros.
    MaxPooling3D(2)	    Vuelve a reducir dimensionalidad, permitiendo aprender jerarquías de representación.
    Dropout(0.3)	    Ayuda a generalizar el modelo.
    Flatten()	        Convierte el volumen 3D resultante en un vector 1D para conectarlo con capas densas (fully connected).
    Dense(128)	        Capa densa con 128 neuronas: aprende combinaciones no lineales complejas de los patrones detectados.
    Dropout(0.4)	    Último control contra overfitting.
    Dense(4)	        Capa de salida softmax, con 4 neuronas (una por clase): devuelve la probabilidad de pertenecer a cada clase.
    """
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn3d(X, y, epochs=30, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = build_cnn3d(input_shape=X.shape[1:], num_classes=y.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, validation_split=0.1,
              epochs=epochs, batch_size=batch_size, callbacks=[early_stop])

    loss, acc = model.evaluate(X_test, y_test)

    print(f"Test accuracy: {acc:.4f}")
    return model
