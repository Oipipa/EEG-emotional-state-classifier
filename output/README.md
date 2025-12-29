Legacy directory kept for backward compatibility with earlier notebooks.

The production pipeline now writes intermediate CSVs to `data/processed/`
and model artifacts to `models/`. You can remove this folder once you
update any Colab notebooks that referenced `/content/drive/.../output`.
