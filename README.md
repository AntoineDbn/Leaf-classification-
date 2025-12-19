IFT712-Projet-Leaf/
│
├── .gitignore              
├── README.md               # Documentation générale
├── requirements.txt        # Liste des librairies
├── main.py                 # A lancer !!!
│
├── data/                   
│   ├── raw/                # Données brutes
│
├── src/                    
│   ├── __init__.py         
│   ├── data_loader.py      # Classe LeafDataLoader
│   ├── models.py           # Modeles du projet 
│   ├── model_trainer.py    # Classe ModelTrainer
│   └── visualization.py    # Fonctions pour tracer les courbes et matrices de confusion
│
├── tests/                  
│   ├── __init__.py
│   └── test_loader.py      # Vérifier que le chargement marche
│
└── rapport/                
    ├── figures/            # Images générées
    └── rapport_final.pdf   # Rapport final
