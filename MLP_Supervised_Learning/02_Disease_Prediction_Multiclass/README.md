# Classification multiclasse : prédiction de la maladie à partir des symptômes

## Objectif

Prédire la maladie d'un patient parmi **30 pathologies** à partir de son âge, de son genre et de la liste de ses symptômes.

## Données

`Healthcare.csv` : 25 000 patients.

- `Age`, `Gender` (Male / Female / Other), `Symptoms` (liste textuelle, par exemple « fever, back pain, shortness of breath »), `Symptom_Count` (de 3 à 7).
- Cible `Disease` : 30 classes équilibrées (de 626 à 729 exemples par classe dans l'entraînement). Exemples : Allergy, Anemia, Anxiety, Stroke, Tuberculosis…

## Contenu du dossier

| Fichier | Rôle |
|---|---|
| `multiclass.ipynb` | Notebook : prétraitement, entraînement, évaluation, diagnostics |
| `Healthcare.csv` | Données brutes |
| `Healthcare_rebuilt.csv` | Données après encodage (33 variables + cible encodée) |

## Démarche

1. **Encodage des symptômes** : chaque liste est découpée puis transformée en vecteur binaire, avec une colonne par symptôme (28 symptômes distincts).
2. **Encodage du genre** : one-hot sur 3 colonnes. La cible est encodée avec `LabelEncoder`.
3. On obtient **33 variables d'entrée** : 2 numériques, 3 pour le genre, 28 pour les symptômes.
4. **Découpage** stratifié 80 / 10 / 10, soit 20 000 / 2 500 / 2 500 patients. `Age` et `Symptom_Count` sont standardisés.
5. **Modèle** : MLP 33 → 256 → 128 → 64 → 30, avec BatchNorm, ReLU et Dropout de 0,1 à chaque couche cachée.
6. **Entraînement** : entropie croisée, Adam (taux 0,001, weight decay 0,0001), mini-lots de 64, early stopping avec une patience de 15 époques.

## Résultats

| Modèle | Accuracy test | F1-macro |
|---|---|---|
| MLP | 3,2 % | 0,028 |
| Régression logistique (référence) | 4,0 % | 0,037 |
| Hasard (1 chance sur 30) | 3,3 % | — |

**Aucun modèle ne fait mieux que le hasard.**

## Diagnostic

Plusieurs vérifications ont été faites pour savoir si le problème venait du code ou des données :

1. **Recherche du taux d'apprentissage** (*LR finder*) : la courbe de loss a été tracée pour des taux allant de 1e-6 à 1.
2. **Contrôle des entrées** : les variables sont bien standardisées et les étiquettes vont bien de 0 à 29.
3. **Test de surapprentissage volontaire** : sur un petit échantillon de 500 patients, le MLP atteint **100 %** d'accuracy en 200 époques.

Le troisième test montre que le code d'entraînement fonctionne : le réseau est capable de mémoriser. Comme ni le MLP ni la régression logistique ne généralisent, la conclusion est que **les variables ne contiennent pas d'information exploitable sur la maladie**. Le jeu de données est très probablement synthétique, avec des symptômes tirés au hasard indépendamment de la maladie.

C'est un résultat négatif, mais instructif : avant d'accuser l'architecture, il faut vérifier qu'il existe un signal dans les données. Un modèle de référence simple et un test de surapprentissage suffisent à le faire.

## Remarque

Les cellules du notebook ont été exécutées dans le désordre : les diagnostics ont été ajoutés après l'entraînement principal. Pour reproduire les résultats, exécuter le notebook de haut en bas.
