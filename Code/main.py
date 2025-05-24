import pandas as pd
import os
from tensorflow.keras.models import load_model
from src.preprocessing.merge_data import merge_features_with_metadata
from src.model.cnn3d import load_data, train_cnn3d
from src.dataset.build_dataset import build_dataset, build_dataset_types
from tensorflow.keras.layers import Conv3D, Dense, MaxPooling3D, Flatten, Dropout
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Variable global para ruta de salida
output_subfolder = ""

def seleccionar_ruta():
    global output_subfolder
    print("Selecciona el tipo de modelo:")
    print("1 - ¿Tiene cáncer?")
    print("2 - Tipo de cáncer")
    tipo = input("Ingresa el número de la opción: ")

    if tipo == "1":
        output_subfolder = "Code/outputs/has_cancer"
    elif tipo == "2":
        output_subfolder = "Code/outputs/identify_type_cancer"
    else:
        print("Opción inválida. Se usará por defecto 'has_cancer'.")
        output_subfolder = "Code/outputs/has_cancer"

def generar_labels():
    print(f"Generando clinical_labels para: {output_subfolder}")
    
    merged_df = merge_features_with_metadata(
        features_path="Data/features.csv",
        metadata_path="Data/metadata.csv"
    )
    os.makedirs(output_subfolder, exist_ok=True)
    merged_df.to_csv(f"{output_subfolder}/features.csv", index=False)

    df = pd.read_csv(f"{output_subfolder}/features.csv")

    if output_subfolder == "Code/outputs/has_cancer":
        df[["PatientID", "has_cancer"]].to_csv(f"{output_subfolder}/clinical_labels.csv", index=False)
        print("✅ clinical_labels.csv (has_cancer) generado correctamente.")

    elif output_subfolder == "Code/outputs/identify_type_cancer":
        # Preprocesar columna Histology
        df["Histology"] = df["Histology"].fillna("no cancer").str.lower().str.strip()

        # Codificación con ColumnTransformer
        ct = ColumnTransformer(
            transformers=[("onehot", OneHotEncoder(sparse_output=False), ["Histology"])],
            remainder="drop"
        )
        one_hot_labels = ct.fit_transform(df[["Histology"]])

        # Guardar etiquetas one-hot junto a PatientID
        patient_ids = df["PatientID"].values.reshape(-1, 1)
        result_array = np.hstack([patient_ids, one_hot_labels])
        result_df = pd.DataFrame(result_array, columns=["PatientID"] + list(ct.named_transformers_["onehot"].get_feature_names_out(["Histology"])))
        result_df.to_csv(f"{output_subfolder}/clinical_labels.csv", index=False)

        # Guardar las clases
        class_names = ct.named_transformers_["onehot"].categories_[0].tolist()
        pd.Series(class_names).to_csv(f"{output_subfolder}/class_names.csv", index=False)

        print(f"✅ clinical_labels.csv y class_names.csv (tipo de cáncer) generados correctamente.")
    else:
        print("❌ No se reconoce el tipo de salida en output_subfolder.")

def construir_dataset():
    base_path = os.path.abspath("Data/NSCLC-Radiomics")

    if(output_subfolder == "Code/outputs/has_cancer"):
        build_dataset(
            base_path=base_path,
            clinical_csv=f"{output_subfolder}/clinical_labels.csv",
            output_dir=output_subfolder,
            shape=(64, 64, 64)
        )
    else:
        build_dataset_types(
            base_path=base_path,
            clinical_csv=f"{output_subfolder}/clinical_labels.csv",
            output_dir=output_subfolder,
            shape=(64, 64, 64)
        )

def entrenar_modelo():
    X_path = f"{output_subfolder}/X.npy"
    y_path = f"{output_subfolder}/y.npy"
    model_path = f"{output_subfolder}/cnn3d_model.keras"

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("❌ Archivos de datos no encontrados. Asegúrate de generar X.npy e y.npy antes de entrenar.")
        return

    print("📥 Cargando datos...")
    X, y = load_data(X_path, y_path)

    print("🧠 Entrenando modelo...")
    model = train_cnn3d(X, y)

    print("💾 Guardando modelo...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    print("✅ Modelo entrenado y guardado en:", model_path)

def mostrar_modelo():
    model_path = f"{output_subfolder}/cnn3d_model.keras"
    model = load_model(model_path)
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
    seleccionar_ruta()

    print("\nSelecciona una opción:")
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
