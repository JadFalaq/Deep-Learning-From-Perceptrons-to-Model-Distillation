# Perceptron multicouche pour trois problèmes supervisés

## Objectif

Développer un perceptron multicouche (MLP) avec PyTorch et l'adapter aux trois grandes familles de problèmes supervisés. Pour chacune, le travail couvre le prétraitement, le choix de la couche de sortie et de la fonction de perte, l'entraînement avec early stopping et l'évaluation avec les métriques adaptées.

Chaque mini-projet est dans son propre sous-dossier, avec son README, ses données et son notebook. Le compte rendu commun est `Développement d’un Perceptron Multicouche (MLP).pdf`.

## Les trois mini-projets

| Dossier | Problème | Sortie / perte | Résultat |
|---|---|---|---|
| [01_Energy_Efficiency_Regression](01_Energy_Efficiency_Regression/) | Régression multi-sortie : charges de chauffage et de climatisation d'un bâtiment | 2 neurones linéaires / MSE | R² de 0,994 (chauffage) et 0,979 (climatisation) |
| [02_Disease_Prediction_Multiclass](02_Disease_Prediction_Multiclass/) | Classification à 30 classes : maladie à partir des symptômes | 30 logits / entropie croisée | 3,2 % d'accuracy, soit le niveau du hasard |
| [03_Bank_Fraud_Detection](03_Bank_Fraud_Detection/) | Classification binaire déséquilibrée : transactions frauduleuses | 1 logit / entropie croisée binaire pondérée | Rappel fraude de 84,5 %, ROC-AUC de 0,90 |

## Éléments communs

- Découpage en trois ensembles entraînement / validation / test, stratifié pour les classifications.
- Normalisation (`StandardScaler`) ajustée uniquement sur l'entraînement, pour éviter toute fuite d'information.
- MLP construits avec des blocs `Linear → ReLU → Dropout`, et BatchNorm pour la classification multiclasse.
- Optimiseur Adam, et **early stopping** sur la loss de validation avec restauration des meilleurs poids.

## Exécution

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch jupyter
```

Chaque notebook lit son CSV dans son propre dossier. Il suffit de l'ouvrir et de l'exécuter de haut en bas.
