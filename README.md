# Predictive-Maintenance-for-Sensor-Data

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
predictive-maintenance/
│
├── README.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/                  # Données originales, jamais modifiées
│   ├── external/             # Données externes
│   ├── interim/              # Données après transformations intermédiaires
│   └── processed/            # Dataset final pour le ML
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_evaluation.ipynb
│
├── models/
│   ├── baseline/
│   ├── production/
│   └── metadata/
│
├── reports/
│   ├── figures/
│   └── results/
│
├── docs/
│
├── references/
│   └── data_dictionary.md
│
├── src/
│   └── predictive_maintenance/
│       ├── __init__.py
│       │
│       ├── config.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ingestion.py
│       │   ├── validation.py
│       │   └── preprocessing.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   └── engineering.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── train.py
│       │   ├── predict.py
│       │   └── evaluate.py
│       │
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── plots.py
│       │
│       └── utils/
│           ├── __init__.py
│           └── helpers.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
└── configs/
    ├── base.yaml
    └── model.yaml
```

--------

