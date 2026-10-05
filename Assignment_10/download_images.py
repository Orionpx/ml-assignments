import pandas as pd
import requests
import os
from PIL import Image

print("Starting image download...")
df = pd.read_csv("multimodal_train.tsv", sep="\t")
df = df.dropna(subset=["clean_title", "image_url"])
os.makedirs("images", exist_ok=True)

success = 0
for row in df.itertuples():
    path = f"images/{row.id}.jpg"
    if not os.path.exists(path):
        try:
            r = requests.get(row.image_url, timeout=5)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    f.write(r.content)
                Image.open(path).convert("RGB")
                success += 1
                print(f"Downloaded: {row.id}.jpg")
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            print(f"Failed to download {row.id}.jpg")

print(f"Finished! Successfully downloaded {success} images.")
