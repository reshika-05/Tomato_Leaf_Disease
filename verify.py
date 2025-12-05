from PIL import Image, ImageFile
import os

# Prevent PIL from freezing on corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

root = r"C:\Users\Dharani\Desktop\Reshi Plant\tomato_diseases"
bad = 0

print("Scanning for corrupted images...\n")

for folder, _, files in os.walk(root):
    for f in files:
        path = os.path.join(folder, f)

        try:
            img = Image.open(path)
            img.load()
            img.close()
        except Exception as e:
            print("Corrupted:", path)
            bad += 1

print("\nScan Complete! Corrupted files found:", bad)