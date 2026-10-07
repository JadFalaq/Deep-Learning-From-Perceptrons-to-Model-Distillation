# Analyse de signaux biomédicaux par convolution (implémentation NumPy)

## Objectif

Comprendre ce que fait réellement une couche convolutive en la codant **à la main avec NumPy**, sans PyTorch ni TensorFlow. Le projet traite trois types de signaux médicaux de dimension croissante : un ECG (1D), une radiographie (2D) et une IRM (3D). Pour chacun, il étudie l'effet du padding, du stride, du pooling, ainsi que les propriétés d'équivariance et d'invariance par translation.

Les signaux sont de petits tableaux définis directement dans le notebook. Cela permet de vérifier chaque résultat à la main.

## Contenu du dossier

| Fichier | Rôle |
|---|---|
| `notebook.ipynb` | Toutes les implémentations et expériences |
| `Analyse_des_signaux_biomedicaux.pdf` | Compte rendu (12 pages) |
| `mission1_ecg.png` | ECG avec les points d'arythmie détectés |
| `conv1d_matin_soir.png` | Équivariance par translation sur deux ECG décalés dans le temps |
| `mission2_sobel_gradient.png` | Sobel-X, Sobel-Y et magnitude du gradient sur la radio |
| `mission2_q8.png`, `mission2_sobel_shift_transpose.png` | Effet d'une translation et d'une transposition de l'image |
| `mission3_coupes_Y.png` | Coupes axiale, coronale et sagittale de la sortie 3D |

## Fonctions implémentées

| Fonction | Description |
|---|---|
| `conv1d(signal, kernel, padding, stride)` | Convolution 1D avec zero-padding et pas réglables |
| `conv2d(image, kernel, padding, stride)` | Convolution 2D (en pratique une corrélation : le noyau n'est pas retourné, comme dans les frameworks de deep learning) |
| `conv3d(volume, kernel, padding, stride)` | Convolution 3D sur un volume |
| `maxpool2d`, `avgpool2d`, `global_avg_pool2d` | Pooling 2D (max, moyenne, moyenne globale) |
| `maxpool3d`, `relu` | Pooling 3D et activation ReLU |
| `detect_arythmie(conv_output, threshold)` | Repère les positions où la sortie de convolution dépasse un seuil |

La taille de sortie suit la formule `(N + 2p − k) / s + 1`, où N est la taille de l'entrée, p le padding, k la taille du noyau et s le pas.

## Mission 1 — ECG (1D)

- Signal de 20 échantillons, filtre `[1, −2, 1]`, qui calcule une dérivée seconde et fait ressortir les variations brusques.
- Taille de la sortie selon la configuration : **18** sans padding, **20** avec padding 1 (même taille que l'entrée), **9** avec un pas de 2.
- **Détection d'arythmie** par le seuil |y| > 3. Le détecteur repère les pics de la zone anormale mais pas toute la zone, et il peut produire un faux positif dans la zone normale. Le filtre réagit en effet aux changements brusques, pas à l'amplitude.
- **Équivariance par translation** : un même battement enregistré le matin et le soir, décalé de 5 échantillons, passe par le filtre `[1, −1]`. Les sorties sont identiques à ce même décalage près : décaler l'entrée décale la sortie d'autant.

## Mission 2 — Radiographie (2D)

- Image 6×6, filtres de Sobel horizontal et vertical (détection de contours).
- Sobel-X sans padding donne une sortie 4×4. Les coefficients `O[0,0] = 16` et `O[1,1] = 14` sont vérifiés à la main dans le notebook.
- **Effet du padding** : avec padding 1, la sortie fait 6×6, mais les coins valent 2 au lieu de 16. Les zéros ajoutés autour de l'image atténuent la réponse sur les bords.
- **Magnitude du gradient** : G = √(Gx² + Gy²) combine les deux directions.
- **Pooling** sur la sortie Sobel-X : le max-pooling 2×2 garde les réponses fortes (25 et −9), l'average-pooling les lisse (16 et −16), et le **global average pooling donne 0**. Les contours positifs et négatifs, symétriques, s'annulent : moyenner sur toute l'image fait perdre cette information.
- **Invariances** :
  - transposer l'image revient à échanger Sobel-X et Sobel-Y : `Sobel-X(A) = Sobel-Y(Aᵀ)ᵀ` ;
  - le **global max pooling** donne la même valeur (25) pour l'image originale, l'image décalée et l'image transposée : il est invariant à ces transformations ;
  - la carte Sobel-X de l'image décalée n'est pas exactement la carte d'origine décalée, à cause des effets de bord.

## Mission 3 — IRM (3D)

- Volume 3×3×3 et noyau 2×2×2 : la sortie fait 2×2×2. La valeur `Y[0,0,0] = 13` est vérifiée à la main.
- Visualisation des trois coupes anatomiques : axiale, coronale et sagittale.
- Mini-pipeline **Conv3D → ReLU → MaxPool3D** : la taille passe de (3,3,3) à (2,2,2) puis à (1,1,1), avec 14 en valeur finale.

## Exécution

```bash
pip install numpy pandas matplotlib jupyter
```

Le notebook est autonome et s'exécute en quelques secondes.
