import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('heart_disease_uci.csv')

df = df.drop('id', axis=1)

df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

valeurs_vides = []