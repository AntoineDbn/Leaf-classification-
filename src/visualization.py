import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, classes, title="Matrice de Confusion (Focus Erreurs)"):
    """
    Affiche une matrice de confusion filtrée.
    Au lieu d'afficher 99x99 cases, on n'affiche que les lignes/colonnes
    où il y a eu des erreurs.
    """
    # 1. Calcul de la matrice complète
    cm = confusion_matrix(y_true, y_pred)
    
    # 2. Identification des indices problématiques
    # On regarde où la diagonale (réussites) est différente de la somme de la ligne (total réel)
    # ou de la somme de la colonne (total prédit)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    diag = cm.diagonal()
    
    # Indices impliqués dans une erreur (soit comme Vraie classe, soit comme Prédiction)
    # Si row_sums[i] != diag[i], ça veut dire qu'il y a des Faux Négatifs pour cette classe
    # Si col_sums[i] != diag[i], ça veut dire qu'il y a des Faux Positifs pour cette classe
    error_indices = np.where((row_sums != diag) | (col_sums != diag))[0]
    
    # 3. Gestion du cas "Modèle Parfait"
    if len(error_indices) == 0:
        print("Incroyable ! Aucune erreur sur le test set. Pas de matrice à afficher.")
        # On crée une image vide pour ne pas faire planter le LaTeX
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Aucune erreur de confusion\nPrécision : 100%", 
                 ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig('rapport/figures/confusion_matrix.png')
        plt.close()
        return

    # 4. Création de la sous-matrice
    # On extrait uniquement les lignes et colonnes intéressantes
    cm_small = cm[np.ix_(error_indices, error_indices)]
    
    # On récupère les noms des espèces correspondantes
    classes_small = [classes[i] for i in error_indices]
    
    # 5. Affichage
    plt.figure(figsize=(10, 8)) # Taille raisonnable
    
    # On utilise un masque pour ne pas colorer les zéros (plus propre)
    mask = (cm_small == 0)
    
    sns.heatmap(cm_small, 
                annot=True,       # Affiche les chiffres
                fmt='d',          # Format entier
                cmap='Reds',      # Rouge pour les erreurs
                mask=mask,        # Cache les cases vides
                xticklabels=classes_small,
                yticklabels=classes_small,
                linewidths=1, 
                linecolor='black',
                cbar=False)       # Pas besoin de barre de couleur pour si peu de valeurs
    
    plt.title(f"{title}\n({len(classes_small)} espèces confondues sur 99)")
    plt.ylabel('Vraie espèce')
    plt.xlabel('Espèce prédite')
    
    # Rotation des étiquettes pour que ce soit lisible
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('rapport/figures/confusion_matrix.png')
    print(f"Matrice sauvegardée (Réduite à {len(classes_small)}x{len(classes_small)} classes).")
    plt.close()

def plot_cv_results(results_df):
    """
    Affiche la comparaison avec barres d'erreur (écart-type).
    """
    plt.figure(figsize=(10, 6))
    
    # On trie pour avoir le meilleur en haut
    df_sorted = results_df.sort_values('best_score', ascending=True)
    
    plt.barh(df_sorted['model_name'], df_sorted['best_score'], 
             xerr=df_sorted['std_score'], capsize=5, 
             color=sns.color_palette("viridis", len(df_sorted)), 
             edgecolor='black')
    
    plt.title('Performance des Modèles (Validation Croisée)')
    plt.xlabel('Précision (Accuracy)')
    plt.xlim(0.5, 1.01) # On laisse un peu de place à droite
    
    # Ligne verticale pour marquer l'excellence
    plt.axvline(x=0.99, color='red', linestyle='--', label='Seuil 99%')
    plt.legend(loc='lower right')
    
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('rapport/figures/model_comparison.png')
    print("Graphique de comparaison sauvegardé.")
    plt.close()