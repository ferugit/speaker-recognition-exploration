import pandas as pd
import matplotlib

df = pd.read_csv('/home/omerhatim/thesis/ok-aura-v1.0.0/dataset.tsv', header = 0, sep = '\t')
df.describe()