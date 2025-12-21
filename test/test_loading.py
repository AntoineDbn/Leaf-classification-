from src.data_loader import LeafDataLoader
import os

train_path = os.path.join('data', 'train.csv') # Assure-toi que tes CSV sont dans un dossier 'data'
test_path = os.path.join('data', 'test.csv')

loader = LeafDataLoader(train_path, test_path)
X, y, X_test, ids = loader.load_data()

print("Exemple de target :", y[:5]) # Devrait afficher des chiffres [0 3 4 ...]
print("Exemple de features :")
print(X.head())
