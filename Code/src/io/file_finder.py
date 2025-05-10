import os

def find_ct_and_seg_paths(patient_study_path: str):
    ct_path = None
    seg_path = None

    for series_folder in os.listdir(patient_study_path):
        full_path = os.path.join(patient_study_path, series_folder)
        if not os.path.isdir(full_path):
            continue

        if "segmentation" in series_folder.lower():
            seg_path = full_path
        else:
            ct_path = full_path if ct_path is None else ct_path

    return ct_path, seg_path
