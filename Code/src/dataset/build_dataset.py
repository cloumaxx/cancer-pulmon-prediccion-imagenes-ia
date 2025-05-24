import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from packaging import version
import sklearn
from src.preprocessing.ct_loader import load_ct_volume

def build_dataset(base_path, clinical_csv, output_dir, shape=(64, 64, 64)):
    labels_df = pd.read_csv(clinical_csv)

    if "PatientID" not in labels_df.columns or "has_cancer" not in labels_df.columns:
        print("Faltan columnas necesarias en el archivo CSV.")
        return

    total = len(labels_df)
    X = []
    y = []

    procesados = 0
    exitosos = 0
    fallidos = 0

    for i, (_, row) in enumerate(labels_df.iterrows(), 1):
        patient_id = row["PatientID"]
        label = row["has_cancer"]

        # Buscar carpeta del paciente
        patient_folders = [d for d in os.listdir(base_path) if d.startswith(patient_id)]
        if not patient_folders:
            print(f"[{i}/{total}] No se encontró carpeta para {patient_id}")
            fallidos += 1
            continue

        patient_dir = os.path.join(base_path, patient_folders[0])

        encontrado = False
        for root, dirs, files in os.walk(patient_dir):
            dicom_files = [f for f in files if f.lower().endswith(".dcm")]
            if dicom_files:
                try:
                    volume = load_ct_volume(root, target_shape=shape)
                    if volume.shape != shape:
                        print(f"[{i}/{total}] Forma incorrecta en {patient_id}: {volume.shape}")
                        fallidos += 1
                        break
                    X.append(volume)
                    y.append(label)
                    exitosos += 1
                    encontrado = True
                    print(f"[{i}/{total}] Procesado {patient_id} ✔️")
                    break
                except Exception as e:
                    print(f"[{i}/{total}] Error al procesar {patient_id}: {e}")
                    fallidos += 1
                    break

        if not encontrado:
            print(f"[{i}/{total}] No se encontró CT válido en {patient_id}")
            fallidos += 1

        procesados += 1

    print(f"\nProcesados: {procesados}, Éxitos: {exitosos}, Fallidos: {fallidos}")

    # Guardar arrays
    if exitosos > 0:
        X = np.stack(X)
        y = np.array(y)
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "X.npy"), X)
        np.save(os.path.join(output_dir, "y.npy"), y)
        print(f"\n✅ Dataset guardado: {X.shape} volúmenes, {len(y)} etiquetas.")
    else:
        print("❌ No se generó ningún volumen válido.")
def build_dataset_types(base_path, clinical_csv, output_dir, shape=(64, 64, 64)):
    labels_df = pd.read_csv(clinical_csv)

    if "PatientID" not in labels_df.columns:
        print("Falta la columna 'PatientID' en el archivo CSV.")
        return

    # 🔍 Detectar columnas one-hot
    label_columns = [col for col in labels_df.columns if col.startswith("Histology_")]
    if not label_columns:
        print("❌ No se encontraron columnas one-hot (tipo Histology_*) en el CSV.")
        return

    total = len(labels_df)
    X = []
    y = []

    procesados = exitosos = fallidos = 0

    for i, (_, row) in enumerate(labels_df.iterrows(), 1):
        patient_id = str(row["PatientID"])
        try:
            label_vector = row[label_columns].astype(float).values
        except Exception as e:
            print(f"[{i}/{total}] ❌ Error en las etiquetas de {patient_id}: {e}")
            fallidos += 1
            continue

        patient_folders = [d for d in os.listdir(base_path) if d.startswith(patient_id)]
        if not patient_folders:
            print(f"[{i}/{total}] 🚫 No se encontró carpeta para {patient_id}")
            fallidos += 1
            continue

        patient_dir = os.path.join(base_path, patient_folders[0])
        encontrado = False

        for root, dirs, files in os.walk(patient_dir):
            dicom_files = [f for f in files if f.lower().endswith(".dcm")]
            if dicom_files:
                try:
                    volume = load_ct_volume(root, target_shape=shape)
                    if volume.shape != shape:
                        print(f"[{i}/{total}] ⚠️ Forma incorrecta en {patient_id}: {volume.shape}")
                        fallidos += 1
                        break
                    X.append(volume)
                    y.append(label_vector)
                    exitosos += 1
                    encontrado = True
                    print(f"[{i}/{total}] ✅ Procesado {patient_id}")
                    break
                except Exception as e:
                    print(f"[{i}/{total}] ❌ Error al procesar {patient_id}: {e}")
                    fallidos += 1
                    break

        if not encontrado:
            print(f"[{i}/{total}] ⚠️ No se encontró CT válido en {patient_id}")
            fallidos += 1

        procesados += 1

    print(f"\n📊 Procesados: {procesados}, Éxitos: {exitosos}, Fallidos: {fallidos}")

    # Guardar los resultados si hubo éxito
    if exitosos > 0:
        X = np.stack(X).astype(np.float32)
        y = np.stack(y).astype(np.float32)

        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "X.npy"), X)
        np.save(os.path.join(output_dir, "y.npy"), y)

        # Guardar nombres de las clases
        pd.Series(label_columns).to_csv(os.path.join(output_dir, "class_names.csv"), index=False)

        print(f"\n✅ Dataset guardado: {X.shape} volúmenes, {y.shape} etiquetas one-hot.")
    else:
        print("❌ No se generó ningún volumen válido.")
