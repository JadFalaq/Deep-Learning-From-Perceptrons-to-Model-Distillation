# Prédiction de maladie cardiaque avec un réseau de neurones

## Objectif

Construire un premier réseau de neurones avec PyTorch pour prédire si un patient est atteint d'une maladie cardiaque à partir de ses données cliniques. Le projet couvre toute la chaîne : exploration des données, nettoyage, encodage, normalisation, entraînement, évaluation et comparaison avec un modèle linéaire.

## Données

Jeu de données **Heart Disease UCI** : 920 patients, 16 colonnes. Il regroupe quatre sources (Cleveland, Hongrie, Suisse, VA Long Beach).

- Variables : âge, sexe, type de douleur thoracique (`cp`), tension au repos (`trestbps`), cholestérol (`chol`), glycémie (`fbs`), ECG au repos (`restecg`), fréquence cardiaque maximale (`thalch`), angine d'effort (`exang`), dépression du segment ST (`oldpeak`), pente ST (`slope`), nombre de vaisseaux colorés (`ca`), thalassémie (`thal`).
- Cible `num` : de 0 (sain) à 4 (gravité croissante). Elle est **binarisée** : 0 = sain, 1 = malade. On obtient 509 malades et 411 sains (55 % / 45 %).

Le fichier `heart_disease_uci.csv` n'est pas inclus. Il est disponible sur [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) et doit être placé dans ce dossier.

## Contenu du dossier

| Fichier | Rôle |
|---|---|
| `heart_disease_mlp.ipynb` | Notebook principal : tout le projet, de l'exploration à l'évaluation |
| `perparation_donnes.ipynb` | Brouillon de préparation (chargement du CSV encodé et normalisation) |
| `pretraitement_initial.py`, `target_enumeration.py` | Premiers scripts de prétraitement (binarisation de la cible) |
| `training_curves.png` | Courbes de loss et d'accuracy pendant l'entraînement |
| `confusion_matrix.png` | Matrice de confusion sur le jeu de test |

Le notebook génère aussi plusieurs CSV intermédiaires : sans valeurs manquantes, imputé, encodé en one-hot, et avec les prédictions.

## Démarche

1. **Valeurs manquantes.** 1 759 valeurs manquent au total. Certaines colonnes sont très incomplètes : `ca` (66 %), `thal` (53 %) et `slope` (34 %). Deux stratégies ont été comparées :
   - supprimer les lignes incomplètes, ce qui ne laisse que 299 patients ;
   - **imputer** par la moyenne (variables numériques) et la valeur la plus fréquente (variables catégorielles), ce qui garde les 920 patients. C'est l'option retenue.
2. **Encodage.** One-hot sur les 8 variables catégorielles (avec `drop_first`). On obtient **21 variables d'entrée**.
3. **Découpage.** 80 % entraînement / 20 % test, stratifié sur la cible (736 / 184 patients).
4. **Normalisation.** `StandardScaler`, ajusté sur le jeu d'entraînement uniquement.
5. **Modèle.** Perceptron multicouche 21 → 32 → 16 → 2 avec activations ReLU (1 266 paramètres).
6. **Entraînement.** Entropie croisée, Adam (taux d'apprentissage 0,001), 100 époques sur le jeu complet (sans mini-lots).

## Résultats

| Modèle | Paramètres | Accuracy test | Remarque |
|---|---|---|---|
| **MLP 32-16** (modèle principal) | 1 266 | **84,8 %** | Rappel de 0,91 sur les malades, 0,77 sur les sains |
| Modèle linéaire (une seule couche) | 44 | 72,8 % | Mêmes hyperparamètres |
| MLP 32-16, taux 0,0001 et 200 époques | 1 266 | 70 % | Sous-apprentissage : ne reconnaît que 38 % des patients sains |
| MLP profond 128-64-32-16 | 13 714 | 85 % | Pas de gain net par rapport au modèle principal |

Rapport de classification du modèle principal :

```
              precision    recall  f1-score   support
   0 (sain)        0.88      0.77      0.82        82
   1 (malade)      0.83      0.91      0.87       102
   accuracy                            0.85       184
```

## Analyse

- Les couches cachées apportent environ 12 points d'accuracy par rapport au modèle linéaire : la relation entre les variables cliniques et la maladie n'est pas linéaire.
- Le modèle détecte mieux les malades (rappel de 0,91) que les sains. C'est le comportement souhaitable en contexte médical, où un faux négatif coûte plus cher qu'un faux positif.
- Approfondir le réseau n'améliore pas le score : avec 920 patients, la quantité de données limite plus la performance que la capacité du modèle.

## Exécution

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch jupyter
```

Placer `heart_disease_uci.csv` dans ce dossier, puis exécuter `heart_disease_mlp.ipynb` de haut en bas.
