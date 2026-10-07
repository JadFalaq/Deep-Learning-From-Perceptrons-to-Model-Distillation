import pandas as pd

df = pd.read_csv('heart_disease_uci.csv')

df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

print(df['target'].value_counts())

df.to_csv('heart_desease_uci_processed.csv', index=False)

