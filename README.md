# Gender Classification from Voice using ML and Django

This project is a machine learning application that predicts the gender (male/female) of a speaker based on vocal properties. The trained model is deployed using a Django web app where users can input feature values and receive instant predictions.

## 🔍 Dataset

The dataset used is [`gender_voice_dataset.csv`](https://www.kaggle.com/datasets/primaryobjects/voicegender), which includes 20 acoustic features extracted from recorded speech samples.

### Target Variable
- `label`: `male` or `female`

### Sample Features
- `meanfreq`, `sd`, `median`, `Q25`, `Q75`, `IQR`, `skew`, `kurt`, `sp.ent`, `sfm`, `mode`, `centroid`, `meanfun`, `minfun`, `maxfun`, `meandom`, `mindom`, `maxdom`, `dfrange`, `modindx`

## 📊 Preprocessing & Feature Engineering

- Null values were checked and none were found.
- Correlation analysis revealed multicollinearity among features.
- Feature selection was performed: some features with high correlation or low predictive power were dropped (`meanfreq`, `kurt`, `sfm`, `meandom`, `dfrange`).

## 📈 Model Building

- Two classifiers were tested:
  - **Support Vector Machine (SVC)**: ~73% accuracy
  - **Random Forest Classifier**: ~98% accuracy

Random Forest showed better generalization and was selected for deployment.

## 🧠 Model Deployment

- The trained model (`voice_model.pickle`) is saved using Python's `pickle` module.
- A **Django** application was developed to:
  - Accept user input for features
  - Load the trained model
  - Predict and return the speaker's gender

## 💡 Features of the Web App

- Simple UI form to enter all required acoustic features.
- Backend processes input using the pre-trained model.
- Displays the predicted gender (`Male` or `Female`) instantly.


