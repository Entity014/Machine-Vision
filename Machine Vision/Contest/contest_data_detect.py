import os
import shutil
import pandas as pd

source_dirs = [
    r"datasets/Questionair Images",
    r"datasets/Intragram Images [Original]",
]
dest_dir = r"datasets/pair"

csv_path = r"data_csv/pair/merged_data.csv"
df = pd.read_csv(csv_path)

image_files = set(df["Image 1"]).union(set(df["Image 2"]))

for src_dir in source_dirs:
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file in image_files:
                src_path = os.path.join(root, file)
                label = df[df["Image 1"] == file]["Menu"].values
                if len(label) == 0:
                    label = df[df["Image 2"] == file]["Menu"].values

                if len(label) > 0:
                    label = label[0]
                else:
                    label = "unknown"

                dest_folder = os.path.join(dest_dir, label)
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)

                dest_path = os.path.join(dest_folder, file)
                shutil.copy2(src_path, dest_path)
                print(f"Copied: {file} to {dest_folder}")

print("File copying completed.")
