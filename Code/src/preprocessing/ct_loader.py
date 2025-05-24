import SimpleITK as sitk
import numpy as np
from skimage.transform import resize

def load_ct_volume(dicom_dir: str, target_shape=(64, 64, 64)) -> np.ndarray:
    # Leer las imágenes DICOM
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Convertir a array numpy (shape: [slices, height, width])
    array = sitk.GetArrayFromImage(image)

    # Normalizar intensidad (ventana de pulmón)
    array = np.clip(array, -1000, 400)
    array = (array - np.min(array)) / (np.max(array) - np.min(array))  # [0, 1]

    # Asegurar que el volumen tenga 3 dimensiones
    if array.ndim != 3:
        raise ValueError(f"Volumen en {dicom_dir} no es 3D, tiene shape {array.shape}")

    # Redimensionar a target_shape (z, y, x)
    resized = resize(array, output_shape=target_shape, mode='constant', preserve_range=True, anti_aliasing=True)

    return resized.astype(np.float32)
