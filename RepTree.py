import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import graphviz
import pydotplus
from six import StringIO

class RepTree:
    """
    Une implémentation de l'algorithme REPTree inspirée de Weka.
    Cette classe utilise scikit-learn's DecisionTreeClassifier comme base.
    L'élagage est géré directement par le paramètre ccp_alpha de DecisionTreeClassifier.
    """
    
    def __init__(self, mode = 'Classification', max_depth=None, min_samples_split=2, ccp_alpha=0.0001):
        """
        Initialise le RepTree avec les paramètres spécifiés.
        
        Args:
            max_depth (int, optional): Profondeur maximale de l'arbre. Defaults to None.
            min_samples_split (int, optional): Nombre minimum d'échantillons requis pour diviser un nœud. Defaults to 2.
            ccp_alpha (float, optional): Paramètre de complexité pour l'élagage. Plus petit = moins d'élagage. Defaults to 0.0001.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha
        if mode == 'Classification':
            self.tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                ccp_alpha=self.ccp_alpha
            )
        elif mode == 'Regression':
            self.tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                ccp_alpha=self.ccp_alpha
            )
        else:
            raise ValueError("Invalid mode. Mode must be 'Classification' or 'Regression'.")

        self.label_encoders = {}
        self.feature_names = None
        self.class_names = None

        
    def fit(self, X, y):
        """
        Entraîne l'arbre de décision sur les données.
        
        Args:
            X: Features d'entraînement
            y: Labels d'entraînement
        """
        # Encodage des variables catégorielles si nécessaire
        X_encoded = X.copy()
        for column in X_encoded.columns:
            if X_encoded[column].dtype == 'object': 
                le = preprocessing.LabelEncoder()
                X_encoded[column] = le.fit_transform(X_encoded[column])
                self.label_encoders[column] = le  
        
        
        # Entraînement de l'arbre
        self.tree.fit(X_encoded, y)
        
        # Sauvegarde des noms de features et de classes
        self.feature_names = X.columns.tolist()
        self.class_names = [str(c) for c in np.unique(y).tolist()]
    
    def predict(self, X):
        """
        Prédit les classes pour de nouvelles données.
        
        Args:
            X: Features à prédire
            
        Returns:
            array: Prédictions
        """
        
        X_encoded = X.copy()
        for column in X_encoded.columns:
            if column in self.label_encoders:  # Vérifier si cette colonne a un encodeur
                le = self.label_encoders[column]
                X_encoded[column] = le.transform(X_encoded[column])  # Transformer
        
        return self.tree.predict(X_encoded)
    
    def score(self, X, y):
        """
        Calcule le score de précision sur les données fournies.
        
        Args:
            X: Features
            y: Labels réels
            
        Returns:
            float: Score de précision
        """
        return metrics.accuracy_score(y, self.predict(X))
    
    def plot_tree(self, output_file='tree', view=True):
        """
        Génère une visualisation de l'arbre de décision.
        
        Args:
            output_file (str): Nom du fichier de sortie (sans extension)
            view (bool): Si True, ouvre automatiquement le fichier généré
            
        Returns:
            graphviz.Source: L'objet graphviz contenant la visualisation
        """
        if self.tree is None:
            raise ValueError("L'arbre n'a pas encore été entraîné. Appelez fit() d'abord.")
        
        dot_data = StringIO()
        
        # Création du DOT data
        export_graphviz(
            self.tree,
            out_file=dot_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            special_characters=True
        )
        
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue().replace('\n',''))  
        graph.write_png(output_file)       
        return graph

if __name__ == "__main__":
    # Chargement des données
    df = pd.read_csv("Support/data/weather.csv")
    
    # Préparation des features et labels
    X = df.drop('play', axis=1)
    y = df['play']
    
    # Création et entraînement du RepTree avec un minimum d'élagage
    rep_tree = RepTree(max_depth=None, ccp_alpha=0.1)
    rep_tree.fit(X, y)
    
    # Visualisation de l'arbre
    rep_tree.plot_tree('meteo-gini.png')
    
    # Prédiction et évaluation
    predictions = rep_tree.predict(X)
    accuracy = rep_tree.score(X, y)
    print(f"Accuracy: {accuracy}")
