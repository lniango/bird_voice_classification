import pandas as pd; 

df = pd.read_csv('annotations.csv'); 
print('Fichiers uniques :', df['Filename'].nunique()); 
print('Annotations connues :', (df['Species eBird Code'] != '????').sum())
