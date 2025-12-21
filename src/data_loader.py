import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

class LeafDataLoader:
    def __init__(self, raw_data_dir='data/raw'):
        # construit les chemins automatiquement à partir du dossier donné
        self.train_path = os.path.join(raw_data_dir, 'train.csv')
        self.test_path = os.path.join(raw_data_dir, 'test.csv')
        self.label_encoder = LabelEncoder()

    def load_data(self):
        """
        Charge les données et sépare features/target.
        Retourne: X_train, y_train, X_test, test_ids, classes
        """
        print(f"Chargement des données depuis : {self.train_path}")
        
        # Vérification de sécurité
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"ERREUR : Le fichier {self.train_path} n'existe pas. Vérifie que le dossier 'data/raw' existe et contient les fichiers csv.")

        df_train = pd.read_csv(self.train_path)
        df_test = pd.read_csv(self.test_path)

        # Séparation Features / Target
        # On drop 'id' (inutile) et 'species' (cible)
        X = df_train.drop(['id', 'species'], axis=1)
        y_raw = df_train['species']

        # Préparation soumission
        test_ids = df_test['id']
        X_test = df_test.drop(['id'], axis=1)

        y = self.label_encoder.fit_transform(y_raw)
        
        return X, y, X_test, test_ids, self.label_encoder.classes_
