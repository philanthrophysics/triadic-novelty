import os
import mvpnovelty.measures as mvm
import pandas as pd
import numpy as np

# locate the dataset relative to this script
data_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "Artificial_Scholarly_Dataset.csv")

# load into a DataFrame
df = pd.read_csv(data_path)

# instantiate and compute pioneer novelty
cd = mvm.CitationData(df,useReferencesSubjects=False)
results = cd.calculatePioneerNoveltyScores()

# print per‐paper pioneer novelty scores
print("Pioneer novelty score per paper:")
