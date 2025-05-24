import pandas as pd

""""
    features_path="data/features.csv",
    metadata_path="data/metadata.csv
"""
def merge_features_with_metadata(features_path, metadata_path):
    # Cargar los archivos
    features_df = pd.read_csv(features_path)
    """
    Columna	            Significado
    PatientID	        Identificador único del paciente, en este caso "LUNG1-001".
    age	                Edad del paciente al momento del diagnóstico, por ejemplo: 78.7515 años.
    clinical.T.Stage	Clasificación del tumor primario (T) según el sistema TNM (T1, T2, T3, T4). En este caso, 2 representa T2.
    Clinical.N.Stage	Afectación de ganglios linfáticos regionales (N). Por ejemplo: 3 puede representar N3.
    Clinical.M.Stage	Presencia de metástasis a distancia (M). 0 indica que no hay metástasis.
    Overall.Stage	    Etapa clínica global del cáncer combinando T, N y M. En este ejemplo: IIIb (una etapa avanzada).
    Histology	        Tipo histológico del cáncer de pulmón. Ejemplos: "large cell", "adenocarcinoma", "squamous cell".
    gender	            Sexo del paciente: "male" o "female".
    Survival.time	    Tiempo de supervivencia en días desde el diagnóstico o inicio del seguimiento hasta la muerte o último control. En este caso: 2165 días (~5.9 años).
    deadstatus.event	Evento de fallecimiento: 1 indica que el paciente murió, 0 indica que aún vive o está censado.
    """
    #features_filter_df = features_df[features_df['Histology'].notna()]
    features_df['has_cancer'] = features_df['Histology'].notna().astype(int)

    return features_df
