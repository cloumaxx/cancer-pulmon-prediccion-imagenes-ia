import SimpleITK as sitk
import numpy as np
from skimage.transform import resize

def load_ct_volume(dicom_dir: str, target_shape=(128, 128, 64)) -> np.ndarray:
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    array = sitk.GetArrayFromImage(image)  # (slices, height, width)
    array = np.clip(array, -1000, 400)  # ventana para pulmón
    array = (array - np.min(array)) / (np.max(array) - np.min(array))  # [0, 1]

    # Resize a forma fija
    resized = resize(array, target_shape, preserve_range=True)
    return resized.astype(np.float32)
