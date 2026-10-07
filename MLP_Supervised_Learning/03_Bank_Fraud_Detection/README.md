# Classification binaire : détection de transactions bancaires frauduleuses

## Objectif

Repérer les transactions suspectes ou frauduleuses dans un historique bancaire. Les classes sont déséquilibrées : il faut donc privilégier le **rappel sur les fraudes** plutôt que l'accuracy globale.

## Données

`Bank_transaction_scenario1.csv` (séparateur `;`) : 5 382 transactions.

- Colonnes utilisées : `Type de transaction` (ATM, paiement en ligne, paiement électronique), `Status operation` (validé, échoué, en attente), `Localisation` (50 villes), `Date`, `Montant`.
- Colonnes supprimées car ce sont des identifiants : `ID Clients`, `Numero de compte`, `Identifiant operation`.
- Cible d'origine : Normal (4 093), Suspect (1 091), Fraude (198). **Suspect et Fraude sont regroupés en une seule classe positive**, soit 1 289 positifs (24 %) contre 4 093 négatifs.

## Contenu du dossier

| Fichier | Rôle |
|---|---|
| `Classification_Binaire_Bank_transaction_anmolie.ipynb` | Notebook : feature engineering, entraînement, évaluation |
| `Bank_transaction_scenario1.csv` | Données brutes |
| `Bank_transaction_scenario1_processed.csv` | Données après feature engineering (62 variables + cible) |

## Démarche

1. **Variables temporelles** extraites de la date : heure, mois, jour de la semaine, indicateur de week-end.
2. **Fréquence de la localisation** (`freq_loc`) : proportion des transactions effectuées dans chaque ville.
3. **One-hot** sur le type de transaction, le statut et la localisation.
4. **Montant** passé au logarithme (`log1p`), car sa distribution est très étalée (de 1 000 à 15 000 000).
5. **Découpage** stratifié 80 / 10 / 10, soit 4 305 / 538 / 539 transactions. Standardisation des variables continues uniquement.
6. **Modèle** : MLP 62 → 128 → 64 → 32 → 1, avec ReLU et Dropout de 0,1.
7. **Gestion du déséquilibre** : `BCEWithLogitsLoss` avec `pos_weight` ≈ 3,18 (rapport négatifs / positifs). Une erreur sur une fraude pèse ainsi environ 3 fois plus.
8. **Entraînement** : Adam (taux 0,001), mini-lots de 64. Early stopping avec une patience de 20 époques, déclenché à l'époque 23.

## Résultats (jeu de test, seuil de décision 0,5)

| Métrique | Valeur |
|---|---|
| Accuracy | 82,9 % |
| **Rappel (fraudes détectées)** | **84,5 %** |
| Précision | 60,2 % |
| F1-score | 0,703 |
| ROC-AUC | 0,903 |

Matrice de confusion :

|  | Prédit normal | Prédit fraude |
|---|---|---|
| **Normal** | 338 | 72 |
| **Fraude** | 20 | 109 |

## Analyse

- Le modèle détecte 109 fraudes sur 129. En contrepartie, il produit 72 fausses alertes. Ce compromis est voulu : manquer une fraude coûte plus cher que vérifier une transaction légitime.
- La ROC-AUC de 0,90 montre que le modèle classe bien les transactions par niveau de risque. Le seuil de 0,5 peut être ajusté selon le coût des fausses alertes.
- La loss de validation remonte dès la 4e époque alors que celle d'entraînement continue de baisser : le surapprentissage arrive vite sur ce petit jeu de données.

## Limites et pistes d'amélioration

- **Restauration des meilleurs poids** : le code sauvegarde `best_state = model.state_dict()` sans copie. Or `state_dict()` renvoie des références vers les paramètres, qui continuent d'être modifiés par l'entraînement. Le modèle évalué correspond donc probablement aux poids de la **dernière** époque et non à ceux de la meilleure. Pour corriger : `best_state = {k: v.clone() for k, v in model.state_dict().items()}`. C'est la méthode déjà utilisée dans les deux autres mini-projets.
- Regrouper « Suspect » et « Fraude » simplifie le problème. Une version à trois classes serait plus fidèle à la réalité métier.
- Il serait utile de choisir le seuil de décision sur le jeu de validation, par exemple pour fixer un rappel cible.
