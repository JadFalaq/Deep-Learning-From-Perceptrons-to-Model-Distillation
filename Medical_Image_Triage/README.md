# Système de triage automatique d'images médicales

## Objectif

Concevoir les briques d'un système d'aide au triage de radiographies thoraciques, en trois tâches de vision par ordinateur :

- **Partie A, classification** : la radiographie est-elle normale ou présente-t-elle une pneumonie ? Comparaison d'un CNN simple avec du transfer learning (VGG16, ResNet50).
- **Partie B, détection** : localiser des zones d'intérêt par boîtes englobantes. Implémentation de l'IoU et du NMS, puis comparaison de Faster R-CNN et YOLOv8n.
- **Partie C, segmentation** : délimiter une région pixel par pixel avec un U-Net, et mesurer l'intérêt des skip connections.

## Contenu du dossier

| Fichier | Rôle |
|---|---|
| `notebook.ipynb` | Les trois parties à la suite (A, puis B, puis C) |
| `Système de Triage Automatique d’Images Médicales.pdf` | Compte rendu (4 pages) |
| `yolov8n.pt` | Poids pré-entraînés de YOLOv8 nano (6 Mo), téléchargés automatiquement par Ultralytics |

Fichiers créés à l'exécution : `saved_splits/` (découpage train/val/test, poids de classes, configuration).

## Données

| Partie | Jeu de données | Obtention |
|---|---|---|
| A et B | **Chest X-Ray Pneumonia** : 5 856 radiographies, dont 4 273 PNEUMONIA et 1 583 NORMAL | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), dossier `chest_xray/` à placer ici |
| C | **Oxford-IIIT Pet** : 7 349 images d'animaux avec masques de segmentation | Téléchargé automatiquement par `tensorflow_datasets` |

Il n'existe pas de masques de segmentation pour les radiographies. La partie C utilise donc Oxford-IIIT Pet pour valider l'architecture U-Net.

---

## Partie A — Classification (TensorFlow/Keras)

### Préparation

- Le **déséquilibre** est de 2,7 PNEUMONIA pour 1 NORMAL. Il est compensé par des poids de classes (`class_weight`) : 1,85 pour NORMAL et 0,69 pour PNEUMONIA.
- Le **découpage** d'origine contient seulement 16 images de validation. Il est remplacé par un découpage stratifié 70 / 15 / 15, soit 4 099 / 878 / 879 images.
- **Prétraitement** : redimensionnement en 224×224, pixels ramenés dans [0, 1] pour le CNN. Pour VGG16 et ResNet50, on applique à la place le `preprocess_input` de chaque réseau.
- **Augmentation légère**, pour rester réaliste sur le plan médical : rotation de ±3 %, translation de 5 %, zoom de 8 %.

### Modèles

| Modèle | Architecture | Paramètres entraînables |
|---|---|---|
| CNN de base | 2 blocs Conv + MaxPool, puis Dense(128), Dropout(0,3), sortie sigmoïde | 25,7 M (presque tous dans la couche dense après le Flatten) |
| VGG16 gelé | Base ImageNet figée, puis GlobalAveragePooling, Dense(128), Dropout, sortie sigmoïde | 65 793 |
| VGG16 fine-tuné | Même modèle, 4 dernières couches dégelées, taux d'apprentissage 1e-5 | — |
| ResNet50 gelé / fine-tuné | Même principe, 10 dernières couches dégelées | — |

Entraînement : Adam, entropie croisée binaire. Early stopping et `ReduceLROnPlateau` sur la loss de validation.

### Résultats mesurés (jeu de test, 879 images)

| Modèle | Accuracy | F1 | Précision | Rappel |
|---|---|---|---|---|
| CNN de base | 93,17 % | 95,28 % | 96,04 % | 94,54 % |
| VGG16 gelé* | 93,29 % | 95,26 % | 98,18 % | 92,51 % |

\* Entraînement arrêté manuellement pendant la 6e époque : le modèle évalué correspond à 5 époques complètes.

Matrice de confusion du CNN de base : 213 vrais négatifs, 606 vrais positifs, 25 faux positifs, **35 faux négatifs**.

Le CNN atteint sa meilleure loss de validation à l'époque 4. La validation devient ensuite instable, avec un début de surapprentissage que l'early stopping contient.

---

## Partie B — Détection (PyTorch, Ultralytics)

### Implémentations à la main

- **IoU** (*Intersection over Union*), testée sur des cas connus : IoU de 0,143 pour deux carrés qui se recouvrent partiellement ; exemples d'IoU élevée (0,69), moyenne (0,09) et nulle.
- **NMS** (*Non-Maximum Suppression*) : sur 4 boîtes candidates, les 2 doublons sont supprimés avec un seuil d'IoU de 0,5. Le notebook visualise les boîtes avant et après.

### Comparaison des détecteurs

Les deux modèles sont pré-entraînés sur COCO. Ils ont été testés sur 20 radiographies de pneumonie, sur CPU.

| Critère | Faster R-CNN (ResNet50-FPN) | YOLOv8n |
|---|---|---|
| Temps d'inférence moyen | 5 881 ms | **243 ms** (24 fois plus rapide) |
| Détections moyennes par image | 1,75 | 1,25 |
| Score de confiance moyen | 0,74 | 0,53 |

L'effet du seuil de confiance a aussi été mesuré pour Faster R-CNN. Le nombre moyen de détections passe de 2,9 au seuil 0,3 à 0,7 au seuil 0,8.

**Limite importante** : ces détecteurs reconnaissent les 80 classes de COCO (personne, voiture…), pas des lésions pulmonaires. Leurs détections sur des radiographies n'ont pas de sens clinique. La comparaison porte uniquement sur la vitesse et le comportement des seuils. Un usage réel exigerait un fine-tuning sur des boîtes médicales annotées.

---

## Partie C — Segmentation U-Net (TensorFlow/Keras)

- **Tâche** : segmentation binaire, animal ou fond, sur des images 128×128.
- **U-Net** : encodeur à 4 niveaux (64, 128, 256, 512 filtres), bottleneck à 1 024 filtres, décodeur symétrique avec convolutions transposées et **skip connections**. Le modèle compte 31 M de paramètres.
- **Perte** : entropie croisée binaire + Dice loss. **Métriques** : Dice score et accuracy pixel à pixel.
- **Ablation** prévue : la même architecture sans skip connections (`build_unet(use_skip=False)`).

---

## État d'avancement

| Élément | État |
|---|---|
| A : préparation des données, CNN de base | Terminé et évalué |
| A : VGG16 gelé | Évalué après 5 époques (entraînement interrompu) |
| A : VGG16 fine-tuné, ResNet50 gelé et fine-tuné, tableau comparatif | Code écrit, **non exécuté** |
| B : IoU, NMS, Faster R-CNN, YOLOv8n | Terminé |
| C : données, architecture U-Net, pertes | Terminé |
| C : entraînement, évaluation, ablation sans skip connections | **Non exécuté** (interrompu au 2e lot) |

Les entraînements ont été interrompus à cause de leur durée : TensorFlow n'utilise pas le GPU sous Windows natif. Une époque de VGG16 prenait environ 20 minutes, et une étape du U-Net environ 16 secondes.

Dans le compte rendu PDF, les valeurs marquées d'un astérisque (VGG16 fine-tuné, ResNet50, U-Net) sont **indicatives et non mesurées**, comme le précise sa note de bas de page.

**Bug à corriger avant de relancer la partie C** : dans la cellule qui définit `TimeHistory`, la ligne `self.epoch_times.append(...)` est coupée en deux (`self.epoch_ti` / `mes.append(...)`). La cellule de la partie A qui définit cette classe est correcte.

## Exécution

```bash
pip install tensorflow tensorflow-datasets torch torchvision ultralytics opencv-python pillow scikit-learn pandas matplotlib jupyter
```

1. Télécharger le dataset Chest X-Ray et placer le dossier `chest_xray/` (avec ses sous-dossiers `train/`, `val/`, `test/`) dans ce dossier.
2. Exécuter les parties dans l'ordre. Chaque partie réimporte ses bibliothèques et peut être lancée séparément.
3. Pour les parties A et C, un GPU est fortement recommandé : WSL2, Linux ou Google Colab.
