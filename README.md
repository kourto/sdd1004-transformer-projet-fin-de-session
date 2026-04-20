# SDD1004 - Projet de fin de session
### Évaluation des biais linguistiques et des raccourcis statistiques dans un Transformer dédié à l’analyse de sentiment


## Guide

Avant d'exécuter du code, que ce soit les fichier `.py` ou `.ipynb`:

### Environnement

Premièrement:
```bash
python -m venv .venv
```

Ensuite si sur **Linux/Mac:**:
```bash
source .venv/bin/activate
```

ou sinon sur **Windows:**:
```bash
.venv\Scripts\activate
```

### Installation

```bash
python -m pip install --upgrade pip
```

```bash
python -m pip install -r requirements.txt
```

## Fichiers du projet

- `SDD1004_-_Yves_Courteau_-_Rapport.pdf` : rapport de ce projet de fin de session.
- `train.py` : script d'entraînement du modèle.
- `evaluation_performance_du_modele.ipynb` : notebook d'évaluation des performances du modèle.
- `analyse.ipynb` : notebook d'analyse des corrélations superficielles et des exemples adversariaux.
- `out/` : dossier de sortie pour les modèles entraînés et les résultats d'évaluation.
- `requirements.txt` : liste des dépendances nécessaires pour exécuter le projet.
- `README.md` : ce même fichier de documentation dans lequel vous vous trouvez.