from src.data.preprocess_dataset import clean_text


# ------------------------------
# TESTS DE CLEAN_TEXT()
# ------------------------------

def test_clean_text_remove_html():
    assert clean_text("<p>Hello</p>") == "Hello"


def test_clean_text_remove_url():
    assert clean_text("Voir https://google.com") == "Voir"


def test_clean_text_remove_email():
    assert clean_text("Contact moi test@gmail.com") == "Contact moi"


def test_clean_text_remove_phone_number():
    assert clean_text("Appelez-moi au 0612345678") == "Appelez-moi au"


def test_clean_text_add_space_after_punctuation():
    assert clean_text("Bonjour.Monde!Ça marche?Oui.") == "Bonjour. Monde! Ça marche? Oui."


def test_clean_text_normalize_spaces():
    assert clean_text("  Bonjour\n\n   tout le monde\t ") == "Bonjour tout le monde"