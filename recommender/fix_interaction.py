import pandas as pd

file_path = '../data/interactions.csv'

df = pd.read_csv(file_path)

first_col = df.columns[1] # 0 is index
df = df[[col for col in df.columns if col != first_col] + [first_col]]

df.to_csv(file_path, index=False)