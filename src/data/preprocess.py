import pandas as pd
import re


RAW_DATA_INPUT_PATH = "../../data/raw/scraped_reviews_boursobank_pages_154_to_290.csv"
PROCESSED_DATA_OUTPUT_PATH = "../../data/processed/clean_reviews_test.parquet"


# ------------------------------
# SUPPRESSION DES DOUBLONS ET DES VALEURS MANQUANTES, ET RÉINITIALISATION DE L'INDEX
# ------------------------------
def remove_duplicates_and_na(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons et les valeurs manquantes, puis réinitialise l'index."""

    return (
        df.drop_duplicates()
          .dropna()
          .reset_index(drop=True)
    )


# ------------------------------
# SUPPRESSION DES COLONNES NON UTILES
# ------------------------------
def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes "pseudo" et "total_reviews"."""

    return df.drop(columns=["pseudo", "total_reviews"])


# ------------------------------
# CONVERSION DES DATES ET FEATURE ENGINEERING TEMPOREL
# ------------------------------
def process_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les dates en type DateTime et crée des variables temporelles (year, month, year_month)."""

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
    text = re.sub(r"\S+@\S+", " ", text)

    # --- numéros de téléphone ---
    text = re.sub(r"\+?\d[\d\s\-]{8,}\d", " ", text)

    # --- espaces après les ".", "!" et "?" ---
    text = re.sub(r"([.!?])(?=\S)", r"\1 ", text)

    # --- espaces et caractères d'espacement (\n, \t, ...) ---
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------
# MAIN
# ------------------------------
def main():
    """Pipeline de prétraitement des données."""

    df = pd.read_csv(RAW_DATA_INPUT_PATH)

    df = remove_duplicates_and_na(df)
    df = remove_unnecessary_columns(df)
    df = process_dates(df)

    df["review"] = df["review"].apply(clean_text)
    df["title"] = df["title"].apply(clean_text)

    df.to_parquet(PROCESSED_DATA_OUTPUT_PATH, index=False)

    #print(df.head())


# ------------------------------
# EXÉCUTION
# ------------------------------
if __name__ == "__main__":
    main()