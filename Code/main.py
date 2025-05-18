import pandas as pd
import os

# Importaciones según cada opción
from tensorflow.keras.models import load_model
from src.preprocessing.merge_data import merge_features_with_metadata
from src.model.cnn3d import load_data, train_cnn3d
from src.dataset.build_dataset import build_dataset
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv3D, Dense, MaxPooling3D, Flatten, Dropout

def generar_labels():
    merged_df = merge_features_with_metadata(
        features_path="data/features.csv",
        metadata_path="data/metadata.csv"
    )
    merged_df.to_csv("outputs/features.csv", index=False)

    df = pd.read_csv("outputs/features.csv")

    def map_histology_to_class(hist):
        hist = hist.lower().strip()
        if hist in ["no cancer", "none"]:
            return 0
        elif "adeno" in hist:
            return 1
        elif "squamous" in hist:
            return 2
        else:
            return 3

    df["CancerClass"] = df["Histology"].apply(map_histology_to_class)
    df[["PatientID", "CancerClass"]].to_csv("data/clinical_labels.csv", index=False)
    print("✅ clinical_labels.csv generado correctamente.")

def construir_dataset():
    build_dataset(
        base_path="../Data/NSCLC-Radiomics",
        clinical_csv="../Data/clinical_labels.csv",
        output_dir="outputs",
        shape=(64, 64, 64)
    )

def entrenar_modelo():
    X, y = load_data("outputs/X.npy", "outputs/y.npy")
    model = train_cnn3d(X, y)
    model.save("outputs/cnn3d_model.keras")
    print("✅ Modelo entrenado y guardado.")

def mostrar_modelo():
    model = load_model("Code\outputs\cnn3d_model.keras")
    print("📊 Estadísticas del modelo:\n")

    total_layers = len(model.layers)
    total_params = model.count_params()
    conv_layers = sum(1 for l in model.layers if isinstance(l, Conv3D))
    dense_layers = sum(1 for l in model.layers if isinstance(l, Dense))
    pool_layers = sum(1 for l in model.layers if isinstance(l, MaxPooling3D))
    dropout_layers = sum(1 for l in model.layers if isinstance(l, Dropout))
    flatten_layers = sum(1 for l in model.layers if isinstance(l, Flatten))

    print(f"🧱 Total de capas: {total_layers}")
    print(f"🔢 Total de parámetros entrenables: {total_params}")
    print(f"🌀 Capas Conv3D: {conv_layers}")
    print(f"🧠 Capas Dense (densas): {dense_layers}")
    print(f"🪣 Capas MaxPooling3D: {pool_layers}")
    print(f"🧽 Capas Dropout: {dropout_layers}")
    print(f"📄 Capas Flatten: {flatten_layers}")

    print("\n📋 Resumen general:")
    model.summary()

def count_dcm_files(base_path: str) -> int:
    count = 0
    for root, dirs, files in os.walk(base_path):
        count += sum(1 for file in files if file.lower().endswith('.dcm'))

    print(f"Total de archivos .dcm encontrados: {count}")
    return count

if __name__ == "__main__":
    print("Selecciona una opción:")
    print("1 - Generar clinical_labels.csv")
    print("2 - Construir dataset")
    print("3 - Entrenar modelo")
    print("4 - Mostrar arquitectura del modelo")
    print("5 - Contar archivos DICOM")
    opcion = input("Ingresa el número de la opción: ")

    if opcion == "1":
        generar_labels()
    elif opcion == "2":
        construir_dataset()
    elif opcion == "3":
        entrenar_modelo()
    elif opcion == "4":
        mostrar_modelo()
    elif opcion == "5":
        ruta = "Data/NSCLC-Radiomics/"
        count_dcm_files(ruta)
    else:
        print("Opción no válida.")
