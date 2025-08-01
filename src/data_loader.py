import os
import shutil
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#setting seed for reproducibilty
random.seed(43)


def split_dataset(
    source_dir="data/raw/PlantVillage",
    dest_dir="data/processed",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # Ensuring destination folders exist
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

    class_names = os.listdir(source_dir)

    print(f"Found {len(class_names)} classes.")
    
    for class_name in tqdm(class_names, desc="Processing classes"):
        class_path = os.path.join(source_dir, class_name)
        images = os.listdir(class_path)

        train_imgs, temp_imgs = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)

        for split_name, split_images in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            split_class_dir = os.path.join(dest_dir, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copyfile(src, dst)

    print("Dataset successfully split and copied")


if __name__ == "__main__":
    split_dataset()
