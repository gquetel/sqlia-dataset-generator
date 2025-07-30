# Script to create "Big-Superviz" a bigger version of the dataset to evaluate the 
# impact of adding more normal samples to the dataset.

import pandas as pd

# path to new normal samples to add to the train set.

fp_new_normal = "/home/gquetel/experiences-results/dataset-generation/big-superviz/dataset.csv"
fp_superviz = "/home/gquetel/experiences-results/dataset-generation/unsupervized-v6/dataset.csv"

df_n = pd.read_csv(fp_new_normal)
df_s = pd.read_csv(fp_superviz)

df_tot = pd.concat([df_n,df_s])
df_tot.to_csv("../big-superviz.csv", index=False)

