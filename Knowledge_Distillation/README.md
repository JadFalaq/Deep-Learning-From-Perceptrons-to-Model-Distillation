# Knowledge Distillation : compresser un modèle pour l'embarqué

## Objectif

Transférer les connaissances d'un grand réseau, le **teacher** (enseignant), vers un réseau minuscule, le **student** (élève), capable de tourner sur une puce IoT ou une caméra embarquée. Le projet implémente et compare quatre familles de distillation :

| Méthode | Ce que le student apprend à imiter |
|---|---|
| **KD classique** (Hinton et al.) | Les probabilités adoucies (*soft labels*) du teacher |
| **Attention Transfer** (AT) | Les cartes d'attention spatiale des couches intermédiaires |
| **FitNets** | Les feature maps intermédiaires, via une couche d'adaptation 1×1 |
| **RKD** (*Relational KD*) | Les relations entre exemples : distances (RKD-D) et angles (RKD-A) |

## Contenu du dossier

| Fichier | Rôle |
|---|---|
| `notebook.ipynb` | Toutes les expériences (partie 1 MNIST, partie 2 CIFAR-10) |
| `Knowledge Distillation.pdf` | Compte rendu (6 pages) |

Figures générées à l'exécution : `soft_labels_8.png`, `exp1_accuracy.png`, `exp2_temperature.png`, `fitnets_featuremaps.png`, `tsne_rkd.png`, `similarity_matrices.png`, `final_comparison.png`.

## Fonction de perte KD

```
L_KD = α · CE(y, z_S) + (1 − α) · T² · KL( softmax(z_T / T) ‖ softmax(z_S / T) )
```

z_T et z_S sont les sorties brutes (logits) du teacher et du student. La température T adoucit les distributions pour faire apparaître les similarités entre classes. Le facteur T² compense la baisse d'amplitude des gradients. Réglage par défaut : T = 4 et α = 0,4.

---

## Partie 1 — MNIST (chiffres 0, 1 et 8) : puce IoT

- **Données** : MNIST réduit aux classes 0, 1 et 8, redimensionné en 32×32. 18 516 images d'entraînement et 3 089 de test.
- **Teacher** : ResNet-50 (poids ImageNet), première couche adaptée à 1 canal et nouvelle tête à 3 classes. 23,5 M de paramètres, 90 Mo.
- **Student** : `MicroCNN` à 3 blocs Conv + BatchNorm + ReLU + MaxPool, puis une couche linéaire. **7 539 paramètres, 29 Ko**, soit 3 118 fois plus petit.

### Résultats

| Expérience | Résultat |
|---|---|
| Student seul / Student + KD (30 époques) | 99,84 % / **99,90 %** |
| Effet de la température, T = 1, 2, 4, 8 (15 époques) | 99,84 / 99,90 / 99,90 / 99,84 % |
| Taille du student : grand (ResNet-18), moyen (MicroCNN), petit (TinyCNN de 1 875 paramètres) | Sans KD → avec KD : 99,90 → 99,97 %, 99,94 → 99,90 %, 99,71 → 99,71 % |
| KD + Attention Transfer (β = 0,1) | 99,84 % |

| Modèle | Taille | Latence (GPU) | Accuracy |
|---|---|---|---|
| Teacher ResNet-50 | 90 Mo | 5,25 ms | 33,96 % |
| Student + KD | 29 Ko | **0,49 ms** | **99,90 %** |

---

## Partie 2 — CIFAR-10 (chat, cerf, chien, cheval) : caméra embarquée

- **Données** : 4 classes d'animaux de CIFAR-10, soit 20 000 images d'entraînement et 4 000 de test. Augmentation par retournement horizontal.
- **Teacher** : VGG-16 (poids ImageNet), nouvelle couche finale à 4 classes. 134 M de paramètres, 512 Mo.
- **Student** : `TinyCNN_CIFAR` à 4 couches convolutives (16, 32, 64, 128 filtres) avec BatchNorm. **130 692 paramètres, 0,5 Mo**, soit 1 027 fois plus petit.
- **FitNets** : couche d'adaptation 1×1 de 64 vers 128 canaux. La feature map du student est alignée sur celle de la couche 9 de VGG-16. Le teacher passe sur CPU pendant cette expérience pour économiser la mémoire du GPU.
- **RKD** : distances deux à deux normalisées par leur moyenne (RKD-D), cosinus des angles sur 64 triplets aléatoires (RKD-A), perte de Huber.

### Résultats

| Méthode | Accuracy test |
|---|---|
| **Student seul** | **81,77 %** |
| Student + KD (T = 4) | 76,53 % |
| FitNets, γ = 0,1 / 0,5 / 1,0 | 75,33 / 68,55 / 75,72 % |
| RKD-D / RKD-A / RKD combiné | 75,05 / 75,20 / 76,85 % |
| Teacher VGG-16 | 30,28 % |

Latence : 4,72 ms pour le teacher contre 0,59 ms pour le student.

Le notebook contient aussi des analyses qualitatives : cartes d'attention du teacher et du student avant et après FitNets, projection t-SNE des représentations, et matrices de similarité cosinus entre les prototypes de classes.

---

## Analyse et limite principale

**Les teachers n'ont pas été entraînés sur la tâche cible.** Dans les deux parties, la couche finale du teacher (et la première couche pour MNIST) est remplacée par une couche aléatoire, sans fine-tuning ensuite. Leur précision est donc proche du hasard : 34 % pour 3 classes, 30 % pour 4 classes.

Cela explique les résultats :

- **Sur MNIST**, la tâche est si simple que le student seul atteint déjà 99,8 %. Les soft labels d'un teacher non entraîné n'apportent rien de mesurable. Les écarts observés (±0,1 point) sont du bruit.
- **Sur CIFAR-10**, toutes les méthodes de distillation font **moins bien** que le student seul, de 5 à 13 points. Le student est poussé à imiter des sorties et des représentations qui ne contiennent pas l'information des classes cibles.

**Amélioration prioritaire** : fine-tuner chaque teacher sur ses classes avant la distillation, en visant plus de 99 % sur MNIST et plus de 90 % sur CIFAR-10, puis relancer les expériences. On devrait alors observer le gain attendu de la KD, surtout sur CIFAR-10.

Le reste des acquis du projet est indépendant de cette limite : les pertes KD, AT, FitNets et RKD sont implémentées et vérifiées, et un student 1 000 à 3 000 fois plus petit s'exécute 8 à 10 fois plus vite.

## Exécution

```bash
pip install torch torchvision scikit-learn matplotlib numpy jupyter
```

MNIST, CIFAR-10 et les poids ImageNet se téléchargent automatiquement dans `./data` et le cache de PyTorch. Un GPU est fortement recommandé : environ 14 Go de mémoire vidéo sont utilisés avec VGG-16. Les résultats ci-dessus ont été obtenus sur GPU CUDA.
