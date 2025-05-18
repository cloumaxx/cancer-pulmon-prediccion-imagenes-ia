import os
import pandas as pd
import numpy as np
from src.io.file_finder import find_ct_and_seg_paths
from src.preprocessing.ct_loader import load_ct_volume

def build_dataset(base_path, clinical_csv, output_dir, shape=(64, 64, 64)):
    labels_df = pd.read_csv(clinical_csv) 
    
    X = []
    y = []

    for _, row in labels_df.iterrows():
        patient_id = row["PatientID"]
        label = row["CancerClass"]

        patient_dir = os.path.join(base_path, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        study_dirs = os.listdir(patient_dir)
        if len(study_dirs) == 0:
            continue

        study_path = os.path.join(patient_dir, study_dirs[0])
        ct_path, seg_path = find_ct_and_seg_paths(study_path)
        if ct_path is None:
            continue

        try:
            volume = load_ct_volume(ct_path, target_shape=shape)
            X.append(volume)
            y.append(label)
        except Exception as e:
            print(f"Error en {patient_id}: {e}")

    X = np.array(X)
    y = np.array(y)

    # Guardar
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    print(f"Dataset guardado: {X.shape} volúmenes, {len(y)} etiquetas.")
