import glob, pandas as pd

files = glob.glob("data/data_test/images/*.jpg")

rows = [{"id": i, "text": f"Flood in Sri Lanka, image {i}", "image_path": f, "label": "real"}
         for i, f in enumerate(files, 1)]

df = pd.DataFrame(rows)
df.to_csv("data/data_test/test_set.csv", index=False)

print("Test CSV saved at data/data_test/test_set.csv")
