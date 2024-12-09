import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

class BayesianClassifier:
    def __init__(self, csv_file, cat_features, cont_features, target):
        """
        Initialise la classe avec les noms des colonnes catégorielles, continues et la cible.

        :param csv_file: Chemin vers le fichier CSV
        :param cat_features: Liste des colonnes catégorielles
        :param cont_features: Liste des colonnes continues
        :param target: Nom de la colonne cible
        """
        self.csv_file = csv_file
        self.cat_features = cat_features
        self.cont_features = cont_features
        self.target = target
        self.label_encoders = {}
        self.model_gaussian = GaussianNB()
        self.model_categorical = CategoricalNB()

    def load_and_prepare_data(self):
        """Charge et prépare les données."""
        # Chargement des données
        self.df = pd.read_csv(self.csv_file)

        # Encodage des colonnes catégorielles
        for column in self.cat_features:
            le = LabelEncoder()
            self.df[column] = le.fit_transform(self.df[column])
            self.label_encoders[column] = le  # Sauvegarde pour décodage futur

        # Extraction des caractéristiques et de la cible
        self.X_cat = self.df[self.cat_features]
        self.X_cont = self.df[self.cont_features]
        self.X_all = self.df[self.cat_features + self.cont_features]
        self.y = self.df[self.target]

    def split_data(self, test_size=0.4, shuffle=False):
        """Divise les données en ensembles d'entraînement et de test."""
        self.X_cat_train, self.X_cat_test, self.X_cont_train, self.X_cont_test, \
        self.X_all_train, self.X_all_test, self.y_train, self.y_test = train_test_split(
            self.X_cat, self.X_cont, self.X_all, self.y, test_size=test_size, shuffle=shuffle
        )

    def train_models(self):
        """Entraîne les modèles GaussianNB et CategoricalNB."""
        self.model_gaussian.fit(self.X_cont_train, self.y_train)
        self.model_categorical.fit(self.X_cat_train, self.y_train)

    def predict_combined(self):
        """
        Calcule les prédictions combinées à partir des deux modèles.

        :return: Prédictions finales
        """
        # Log-probabilités des modèles individuels
        log_proba_cont = self.model_gaussian.predict_log_proba(self.X_cont_test)
        log_proba_cat = self.model_categorical.predict_log_proba(self.X_cat_test)

        # Log-probabilité combinée
        log_proba_combined = log_proba_cont + log_proba_cat - np.log(self.model_gaussian.class_prior_)

        # Prédictions finales
        return np.argmax(log_proba_combined, axis=1)

    def evaluate(self):
        """Évalue les performances du modèle combiné."""
        y_pred_combined = self.predict_combined()
        accuracy = (y_pred_combined == self.y_test).mean()
        return accuracy

