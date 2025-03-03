import pandas as pd
import numpy as np

file_intragram = "data_csv/temp/data_from_intragram.csv"
df_intragram = pd.read_csv(file_intragram)
df_intragram = df_intragram.iloc[:, :4]

swap_mask = np.random.rand(len(df_intragram)) < 0.5
if "Image 1" in df_intragram.columns and "Image 2" in df_intragram.columns:
    df_intragram.loc[swap_mask, ["Image 1", "Image 2"]] = df_intragram.loc[
        swap_mask, ["Image 2", "Image 1"]
    ].to_numpy()
    df_intragram["Winner"] = np.where(swap_mask, 2, 1)

file1 = "data_csv/temp/data_from_questionaire.csv"
df1 = pd.read_csv(file1)

merged_df = pd.concat([df1, df_intragram], ignore_index=True)

merged_df.to_csv("data_csv/pair/merged_data.csv", index=False)

labels = merged_df["Menu"].unique()
for i, label in enumerate(labels):
    df_group = merged_df[merged_df["Menu"] == label]

    output_file = f"data_csv/pair/data_{label}.csv"
    df_group.to_csv(output_file, index=False)
    print(f"Saved {output_file}")

print("CSV splitting completed.")
