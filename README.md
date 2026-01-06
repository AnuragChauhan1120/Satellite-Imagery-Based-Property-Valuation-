--------------House Price Prediction using Tabular Data and Satellite Imagery-----------------

This project implements a multimodal machine learning pipeline to predict house prices by combining:
Structured tabular features (e.g., size, location, amenities)
Satellite imagery capturing neighborhood-level context

The system uses:
A Convolutional Neural Network (CNN) for image feature extraction
XGBoost for final price prediction
SHAP and Grad-CAM for interpretability

house_price_multimodal/
│
├── data/
│   ├── train_clean.csv
│   ├── test.xlsx
│   ├── train.xlsx
│   ├── test_clean.csv
│   ├── test_predictions.csv
│   └──XGB_CNN_Combined predictions.csv
│
├── satellite_images/
│   ├── 00000.jpg
│   ├── 00001.jpg
│   └── ...
│
├── dependables/
│   ├── cnn_image_model.keras
│   └── feature_CNN.csv
│
├── CNN.py
├── data_fetcher.py
├── preprocessing.py
├── GRAD CAM + SHAP.py
├── Training.py
├── xgb_cnn_tabular.py
│
└── README.md

--------------How to Run the Project (Step-by-Step)---------------

First do : pip install -r requirements.txt
(Install all the dependencies and libraries used in the project)

Important: Run files in the order below.
Each stage produces artifacts used by the next stage.
1- Preprocessing.py 
(It cleans and creates train_clean over which the other codes are processed)

2- data_fetcher.py
(Fetches the satellite images used)

3- CNN.py
(Creates the CNN used within the model)

4- CNN_features.py
(Creates features or embeddings to be used with the tabular data)

5- Training.py
(Trains the CNN+Tabular and Tabular only over the train_clean data)

6- GRAD CAM + SHAP.py
(Explains the contribution of CNN and tabular data in the model trained)





