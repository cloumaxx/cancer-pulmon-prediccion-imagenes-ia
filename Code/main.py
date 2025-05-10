from tensorflow.keras.models import load_model

from src.preprocessing.merge_data import merge_features_with_metadata
from src.model.cnn3d import load_data, train_cnn3d

"""merged_df = merge_features_with_metadata(
    features_path="data/features.csv",
    metadata_path="data/metadata.csv"
)

# Guardar para usar en entrenamiento
merged_df.to_csv("outputs/features.csv", index=False)
"""

from src.dataset.build_dataset import build_dataset
import pandas as pd

# Cargar el archivo
"""df = pd.read_csv("outputs/features.csv")  # Ajusta la ruta si es diferente

# Mapeo de histología a clases
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

# Aplicar el mapeo
df["CancerClass"] = df["Histology"].apply(map_histology_to_class)

# Guardar el archivo con las dos columnas necesarias
df[["PatientID", "CancerClass"]].to_csv("data/clinical_labels.csv", index=False)
print("✅ clinical_labels.csv generado correctamente.")
"""
if __name__ == "__main__":
    # build dataset
    """build_dataset(
        base_path="../Data/NSCLC-Radiomics",
        clinical_csv="../Data/clinical_labels.csv",
        output_dir="outputs",
        shape=(64, 64, 64)
    )"""

    #X, y = load_data("outputs/X.npy", "outputs/y.npy")
    #model = train_cnn3d(X, y)
    #model.save("outputs/cnn3d_model.keras")

    model = load_model("outputs/cnn3d_model.keras")
    model.summary()

