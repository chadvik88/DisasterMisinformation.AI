import os

folders = [
    "data/data_raw", "data/data_processed", "notebooks",
    "src/data_collection", "src/text_model", "src/image_model",
    "src/gnn_model", "src/utils", "scripts", "app"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

open("requirements.txt", "a").close()
open(".gitignore", "a").close()
open("README.md", "a").close()
open(".env", "a").close()
