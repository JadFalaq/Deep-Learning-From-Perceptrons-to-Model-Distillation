# Deep Learning: From Perceptrons to Model Distillation

**Auteur :** Jad Falaq

Les cinq dossiers suivent une progression : réseaux de neurones sur données tabulaires, fonctionnement interne des convolutions, vision par ordinateur appliquée à l'imagerie médicale, puis compression de modèles pour l'embarqué. Chaque dossier contient son propre README qui détaille l'objectif, les données, la démarche, les résultats et les limites.

## Vue d'ensemble

| Projet                                                  | Sujet                                                                                                                   | Outils                                 | Résultat principal                                                                         |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------ |
| [Heart_Disease_MLP](Heart_Disease_MLP/)                 | Prédire une maladie cardiaque à partir de données cliniques                                                             | PyTorch, scikit-learn                  | 84,8 % d'accuracy en test (contre 72,8 % pour un modèle linéaire)                          |
| [MLP_Supervised_Learning](MLP_Supervised_Learning/)     | Trois problèmes supervisés avec un perceptron multicouche : régression, classification multiclasse, détection de fraude | PyTorch, scikit-learn                  | R² de 0,994 et 0,979 en régression ; ROC-AUC de 0,90 sur la fraude                         |
| [Convolutions_From_Scratch](Convolutions_From_Scratch/) | Convolutions 1D, 2D et 3D codées à la main sur des signaux biomédicaux (ECG, radio, IRM)                                | NumPy                                  | Implémentation complète de la convolution, du padding, du stride et du pooling             |
| [Medical_Image_Triage](Medical_Image_Triage/)           | Triage de radiographies : classification de pneumonie, détection d'objets, segmentation                                 | TensorFlow/Keras, PyTorch, Ultralytics | CNN de base à 93,2 % et VGG16 gelé à 93,3 % ; YOLOv8n 24 fois plus rapide que Faster R-CNN |
| [Knowledge_Distillation](Knowledge_Distillation/)       | Distillation de connaissances vers de petits modèles embarquables (KD, Attention Transfer, FitNets, RKD)                | PyTorch                                | Modèle de 7 539 paramètres à 99,9 % sur MNIST (3 classes)                                  |

## État d'avancement

| Projet                    | État                                                                                                                                                                          |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Heart_Disease_MLP         | Terminé                                                                                                                                                                       |
| MLP_Supervised_Learning   | Terminé. Le modèle multiclasse ne dépasse pas le hasard : le README explique pourquoi                                                                                         |
| Convolutions_From_Scratch | Terminé                                                                                                                                                                       |
| Medical_Image_Triage      | Partiel : la classification (CNN, VGG16 gelé) et la détection sont faites ; le fine-tuning de VGG16, ResNet50 et l'entraînement du U-Net n'ont pas été exécutés jusqu'au bout |
| Knowledge_Distillation    | Terminé, avec une limite importante sur les modèles enseignants (voir son README)                                                                                             |

## Installation

Python 3.10 ou plus récent est recommandé.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install numpy pandas matplotlib seaborn scikit-learn jupyter
pip install torch torchvision          # tous les projets sauf Convolutions_From_Scratch
pip install tensorflow tensorflow-datasets opencv-python pillow ultralytics   # Medical_Image_Triage
```

Remarque : TensorFlow n'utilise plus le GPU sous Windows natif depuis la version 2.11. Les entraînements de Medical_Image_Triage sont donc très lents sur Windows. Il vaut mieux utiliser WSL2 ou Google Colab. Knowledge_Distillation a été exécuté sur GPU (CUDA).

## Données

Les jeux de données de MLP_Supervised_Learning sont inclus. Ceux des autres projets ne le sont pas, soit à cause de leur taille, soit parce qu'ils se téléchargent automatiquement.

| Projet                    | Jeu de données                                      | Comment l'obtenir                                                                                                    |
| ------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Heart_Disease_MLP         | Heart Disease UCI (`heart_disease_uci.csv`)         | [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data), à placer dans le dossier du projet      |
| MLP_Supervised_Learning   | Energy Efficiency, Healthcare, Bank transactions    | Inclus dans chaque sous-dossier                                                                                      |
| Convolutions_From_Scratch | Signaux jouets définis directement dans le notebook | Aucun téléchargement                                                                                                 |
| Medical_Image_Triage      | Chest X-Ray Pneumonia (dossier `chest_xray/`)       | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), à placer dans le dossier du projet |
| Medical_Image_Triage      | Oxford-IIIT Pet                                     | Téléchargé automatiquement par `tensorflow_datasets`                                                                 |
| Knowledge_Distillation    | MNIST et CIFAR-10                                   | Téléchargés automatiquement par `torchvision`                                                                        |

## Structure

```
.
├── README.md
├── Heart_Disease_MLP/
├── MLP_Supervised_Learning/
│   ├── 01_Energy_Efficiency_Regression/
│   ├── 02_Disease_Prediction_Multiclass/
│   └── 03_Bank_Fraud_Detection/
├── Convolutions_From_Scratch/
├── Medical_Image_Triage/
└── Knowledge_Distillation/
```
