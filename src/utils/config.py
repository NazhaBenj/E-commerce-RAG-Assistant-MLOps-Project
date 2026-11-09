
from pathlib import Path
import yaml


# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Chemin du fichier de configuration
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_config():
    """
    Charge les paramètres du projet depuis params.yaml.
    """
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {PARAMS_PATH}"
        )

    with open(PARAMS_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Le fichier params.yaml doit contenir un dictionnaire YAML.")

    return config


def get_config(section=None):
    """
    Retourne toute la configuration ou une section spécifique.
    """
    config = load_config()

    if section is None:
        return config

    if section not in config:
        raise KeyError(f"Section '{section}' absente de params.yaml")

    return config[section]


if __name__ == "__main__":
    configuration = load_config()

    print("Configuration chargee avec succes !")
    print("\nParametres du projet :")
    print(configuration)
