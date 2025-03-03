import os
import shutil
import random

# กำหนด path ของ dataset
dataset_dir = "datasets/Intragram Images [Original]"

# กำหนดเส้นทางที่ต้องการเก็บข้อมูล train, validation, test
base_dir = "datasets/class"

# สร้าง directories สำหรับ train, validation, test
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(base_dir, split), exist_ok=True)
    for category in ["Burger", "Dessert", "Pizza", "Ramen", "Sushi"]:
        os.makedirs(os.path.join(base_dir, split, category), exist_ok=True)


# ฟังก์ชั่นเพื่อแบ่งข้อมูล
def split_data(category):
    category_path = os.path.join(dataset_dir, category)
    all_files = os.listdir(category_path)
    random.shuffle(all_files)

    # คำนวณจำนวนไฟล์สำหรับแต่ละชุดข้อมูล
    total_files = len(all_files)
    train_end = int(0.7 * total_files)
    val_end = int(0.85 * total_files)

    # แบ่งไฟล์
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    # ย้ายไฟล์ไปยังโฟลเดอร์ที่เหมาะสม
    for file in train_files:
        shutil.copy(
            os.path.join(category_path, file),
            os.path.join(base_dir, "train", category, file),
        )
    for file in val_files:
        shutil.copy(
            os.path.join(category_path, file),
            os.path.join(base_dir, "val", category, file),
        )
    for file in test_files:
        shutil.copy(
            os.path.join(category_path, file),
            os.path.join(base_dir, "test", category, file),
        )


# แบ่งข้อมูลในแต่ละ category
categories = ["Burger", "Dessert", "Pizza", "Ramen", "Sushi"]
for category in categories:
    split_data(category)

print("Data split completed!")
