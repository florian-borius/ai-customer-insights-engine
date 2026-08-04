import re

import pandas as pd

from config.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
)


# ------------------------------
# SUPPRESSION DES DOUBLONS, DES CHAÎNES VIDES ET DES VALEURS MANQUANTES, ET RÉINITIALISATION DE L'INDEX
# ------------------------------
def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons, les chaînes vides et les valeurs manquantes, puis réinitialise l'index."""

    return (
        df.drop_duplicates()
          .replace(r"^\s*$", pd.NA, regex=True)
          .dropna()
          .reset_index(drop=True)
    )


# ------------------------------
# SUPPRESSION DES COLONNES NON UTILES
# ------------------------------
def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes "pseudo" et "total_reviews"."""

    return df.drop(columns=["pseudo", "total_reviews"], errors="ignore")


# ------------------------------
# CONVERSION DES DATES ET FEATURE ENGINEERING TEMPOREL
# ------------------------------
def process_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les dates en type DateTime et crée des variables temporelles (year, month, year_month)."""

    # Copie du DataFrame pour éviter de modifier celui passé en argument
    df = df.copy()

    mois_fr_en = {
        'janvier': 'January', 'février': 'February', 'mars': 'March', 'avril': 'April',
        'mai': 'May', 'juin': 'June', 'juillet': 'July', 'août': 'August',
        'septembre': 'September', 'octobre': 'October', 'novembre': 'November', 'décembre': 'December'
    }

    # --- conversion de "experience_date" ---
    df["experience_date"] = df["experience_date"].str.lower().replace(mois_fr_en, regex=True)
    df["experience_date"] = pd.to_datetime(df["experience_date"], errors="coerce")

    # --- conversion de "publication_date" ---
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce").dt.tz_convert('Europe/Paris')

    # --- feature engineering temporel ---
    df["year"] = df["publication_date"].dt.year
    df["month"] = df["publication_date"].dt.month
    df["year_month"] = df["publication_date"].dt.strftime("%Y-%m")

    return df


# ------------------------------
# NETTOYAGE DE TEXTE
# ------------------------------
def clean_text(text: str) -> str:
    """
    Nettoie un texte en supprimant les balises HTML, les URLs et les données personnelles (emails, numéros de téléphone),
    en ajoutant un espace après ".", "!" et "?" s'il n'y en a pas,
    et en normalisant les espaces.
    """

    # --- balises HTML ---
    text = re.sub(r"<.*?>", " ", text)

    # --- URLs ---
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # --- emails ---
    text = re.sub(r"\S*@\S+", " ", text)

    # --- numéros de téléphone ---
    text = re.sub(r"\+?\d[\d\s\-.]{8,}\d", " ", text)

    # --- ajout d'un espace après les ".", "!" et "?" s'il n'y en a pas ---
    text = re.sub(r"([.!?]+)(?![.!?])(?=\S)", r"\1 ", text)

    # --- suppression des espaces en trop et des caractères d'espacement (\n, \t, ...) ---
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------
# PIPELINE DE PRÉTRAITEMENT D'UN DATASET
# ------------------------------
def main(input_path: str, output_path: str):
    """Pipeline de prétraitement d'un dataset, avec ajout d'un review_id."""

    df = pd.read_csv(input_path)

    df["review_id"] = df.index

    df = remove_invalid_rows(df)
    df = remove_unnecessary_columns(df)
    df = process_datetime_features(df)
    df["review"] = df["review"].apply(clean_text)
    df["title"] = df["title"].apply(clean_text)
    df = remove_invalid_rows(df)

    df.to_parquet(output_path, index=False)


# ------------------------------
# EXÉCUTION
# ------------------------------
if __name__ == "__main__":
    main(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH)