import csv
import random

titles = [
    "Local man discovers new species of bird in backyard",
    "Shocking truth about the moon landing revealed",
    "Scientists find cure for common cold in rare plant",
    "Politician caught in massive corruption scandal",
    "New study shows chocolate helps you live longer",
    "Aliens have landed in rural farm, farmer claims",
    "Tech company releases revolutionary new gadget",
    "Historical artifact found under city streets",
    "Celebrity spotted doing normal things at grocery store",
    "The world is actually flat, says new controversial group"
]

with open('Assignment_10/multimodal_train.tsv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['id', 'clean_title', 'image_url', '2_way_label'])
    
    for i in range(1, 101):
        clean_title = random.choice(titles) + f" {i}"
        image_url = f"https://picsum.photos/seed/{i}/200"
        label = random.choice([0, 1])
        writer.writerow([f"fake_id_{i}", clean_title, image_url, label])

print("Generated dummy multimodal_train.tsv with working image URLs.")
