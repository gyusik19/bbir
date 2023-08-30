import os
import shutil
import random

random.seed(42)

# Define source and target directories
source_dir = 'data/train2017'
training_dir = 'data/train'
gallery_dir = 'data/gallery'

# Create target directories if they don't exist
os.makedirs(training_dir, exist_ok=True)
os.makedirs(gallery_dir, exist_ok=True)

# List all images in the source directory
all_images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Shuffle the list of images
random.shuffle(all_images)

# Split the list of images
training_images = all_images[:50000]
gallery_images = all_images[50000:]

# Move the images to the respective directories
for img in training_images:
    shutil.move(os.path.join(source_dir, img), os.path.join(training_dir, img))

for img in gallery_images:
    shutil.move(os.path.join(source_dir, img), os.path.join(gallery_dir, img))

print(f"Moved {len(training_images)} images to {training_dir}")
print(f"Moved {len(gallery_images)} images to {gallery_dir}")