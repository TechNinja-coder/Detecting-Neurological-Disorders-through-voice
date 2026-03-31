# Parkinson's Detection Project
This is a project that is trying to see if neuroligical disorders could be analyzed through voice. This project is not made to be a replacement of medical diagnosis and only is the beginning, please do not base your diagnosis on this project.
The notebook is the center of this project.

## Layout

- `src/Parkinsons.ipynb`: canonical training and experimentation workflow
- `src/Parkinsons.py`: shared support code used by the notebook and app
- `src/app.py`: Streamlit interface for inference
- `data/raw/`: input datasets
- `data/results/`: generated analysis outputs
- `models/`: trained model artifacts used by the app

## Workflow

1. Run `src/Parkinsons.ipynb` to load data, train the model, evaluate it, and save artifacts.
2. Launch the UI from the notebook's final Streamlit cell, or run `streamlit run src/app.py`.
3. The app reads the saved artifacts from `models/`.



