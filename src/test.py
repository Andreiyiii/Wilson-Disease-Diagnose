import pandas as pd

df = pd.read_csv("data/Wilson_disease_dataset.csv")
y = df["Is_Wilson_Disease"]

for col in df.columns.drop("Is_Wilson_Disease"):
    same = (df[col] == y).mean()
    if same > 0.55:
        print(col, same)