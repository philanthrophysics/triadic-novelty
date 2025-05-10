import os
import mvpnovelty.measures as mvm
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


# locate the dataset relative to this script
data_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "Artificial_Scholarly_Dataset.csv")

# load into a DataFrame
df = pd.read_csv(data_path)

# instantiate and compute pioneer novelty
# need to have columns: 'publicationID', 'references', 'subjects', 'year'
cd = mvm.CitationData(df)

# calculate the novelty scores for each of the three measures
# pioneer, maverick, and vanguard
df_pioneer = cd.calculatePioneerNoveltyScores()
df_maverick = cd.calculateMaverickNoveltyScores()
df_vanguard = cd.calculateVanguardNoveltyScores()


# merge the results into a single DataFrame They all have the same publicationID in common
df_novelties = df_pioneer.merge(df_maverick, on="publicationID", how="left")
df_novelties = df_novelties.merge(df_vanguard, on="publicationID", how="left")


# We can now calculate the normalized scores
model_novelties_list = []
for i in tqdm(range(100), desc="Generating base model instances"):
    cd_model = cd.generateBaseModelInstance(attractiveness=1.5)
    df_model_pioneer = cd_model.calculatePioneerNoveltyScores()
    df_model_maverick = cd_model.calculateMaverickNoveltyScores()
    df_model_vanguard = cd_model.calculateVanguardNoveltyScores()
    # merge the results into a single DataFrame They all have the same publicationID in common
    df_model_novelties = df_model_pioneer.merge(df_model_maverick, on="publicationID", how="left")
    df_model_novelties = df_model_novelties.merge(df_model_vanguard, on="publicationID", how="left")
    # You can for instance save the results to a feather file to load them later
    # df_model_novelties.to_feather(f"model_novelties_{i}.feather")
    # or append them to a list
    model_novelties_list.append(df_model_novelties)


# We can then create an aggregated list for the base model instances with the mean and std
# (only numeric columns)
# only for the numeric columns
numeric_cols = df_model_novelties.select_dtypes(include=[np.number]).columns.tolist()
df_model_novelties_agg = pd.concat(model_novelties_list).groupby("publicationID").agg(
    {col: ["mean", "std"] for col in numeric_cols}
)

# now calculate the normalized scores (original - mean_model) / std_model
df_novelties_normalized = df_novelties.copy()
for col in numeric_cols:
    df_novelties_normalized[col] = (
        (df_novelties[col] - df_model_novelties_agg[col]["mean"]) / df_model_novelties_agg[col]["std"]
    )




