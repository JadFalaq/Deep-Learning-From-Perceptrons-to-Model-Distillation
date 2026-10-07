# Régression : consommation énergétique des bâtiments

## Objectif

Prédire **simultanément** deux grandeurs à partir de la géométrie d'un bâtiment : sa charge de chauffage (`Heating_Load`) et sa charge de climatisation (`Cooling_Load`). Il s'agit d'une régression multi-sortie.

## Données

`energy_efficiency_data.csv` : jeu de données *Energy Efficiency* (UCI). Il contient 768 bâtiments simulés et aucune valeur manquante.

- **8 variables d'entrée** : compacité relative, surface totale, surface des murs, surface du toit, hauteur, orientation, surface vitrée, répartition du vitrage.
- **2 cibles** : `Heating_Load` et `Cooling_Load`, très corrélées entre elles (r = 0,98), ce qui justifie un seul modèle à deux sorties.

## Contenu du dossier

| Fichier | Rôle |
|---|---|
| `regression.ipynb` | Notebook : exploration, entraînement, évaluation |
| `energy_efficiency_data.csv` | Données brutes |
| `scaled_energy_efficiency_data.csv` | Version standardisée des données (non utilisée par la version actuelle du notebook) |

## Démarche

1. **Exploration** : boîtes à moustaches, histogrammes et matrice de corrélation.
2. **Découpage** : 80 % entraînement / 20 % test, puis 15 % de l'entraînement mis de côté pour la validation (521 / 93 / 154 bâtiments).
3. **Normalisation** : standardisation des entrées **et des cibles**, ajustée sur l'entraînement.
4. **Modèle** : MLP 8 → 64 → 32 → 2, avec ReLU et Dropout de 0,1. La sortie est linéaire car c'est une régression.
5. **Entraînement** : perte MSE, Adam (taux 0,001, weight decay 0,0001), mini-lots de 32. Early stopping avec une patience de 20 époques (500 époques maximum).

## Résultats (jeu de test)

| Cible | MAE | RMSE | R² |
|---|---|---|---|
| Heating_Load | 0,059 | 0,078 | **0,994** |
| Cooling_Load | 0,106 | 0,146 | **0,979** |

Les cibles étant standardisées, MAE et RMSE s'expriment en écarts-types. Par exemple, une MAE de 0,059 sur le chauffage correspond à environ 0,6 kWh/m², puisque l'écart-type de cette cible est de 10,1.

Le notebook trace aussi les courbes d'apprentissage, les prédictions comparées aux valeurs réelles, et la distribution des résidus.

## Analyse

- Le modèle explique plus de 97 % de la variance des deux cibles. Les courbes d'entraînement et de validation restent proches : il n'y a pas de surapprentissage.
- La climatisation est un peu moins bien prédite : elle dépend de phénomènes thermiques plus complexes.
- **Limite** : les données sont issues d'une simulation et sont très structurées. Sur des mesures réelles, plus bruitées, une validation croisée et davantage de régularisation seraient nécessaires.
