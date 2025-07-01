from PIL import Image
import pandas as pd
import os

# Create test directory
os.makedirs("data/data_test/images", exist_ok=True)

# Colors and captions
colors = ["red", "blue", "green", "orange", "purple"]
captions = [
    "Wildfire in California spreading rapidly.",
    "Flooding in Jakarta displaces thousands.",
    "Collapsed buildings after earthquake in Japan.",
    "Heavy snowfall cuts off roads in Kashmir.",
    "Explosion aftermath in downtown Beirut."
]

rows = []

# Generate images + CSV rows
for i, (color, caption) in enumerate(zip(colors, captions)):
    img_path = f"data/data_test/images/test_img_{i}.jpg"
    Image.new("RGB", (256, 256), color=color).save(img_path)

    rows.append({
        "id": i,
        "text": caption,
        "image_path": img_path,
        "label": "real"
    })

df = pd.DataFrame(rows)
df.to_csv("data/data_test/test_set.csv", index=False)

print("Dummy disaster test data created!")
