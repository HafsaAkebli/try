import os
import shutil

# Define the paths
source_dir = "/home/akebli/test5/patches"
destination_base = "/home/akebli/test5/Patches1"

# Define the classes
classes = ["G3", "G4", "Stroma", "Normal", "G5"]

# Ensure that class subfolders exist within each subset's train directory
for subset in ["Subset1", "Subset2", "Subset3"]:
    train_dir = os.path.join(destination_base, subset, "train")
    for cls in classes:
        dest_dir = os.path.join(train_dir, cls)
        os.makedirs(dest_dir, exist_ok=True)

# Function to determine the subset from the filename
def determine_subset(filename):
    if "Subset1" in filename:
        return "Subset1"
    elif "Subset2" in filename:
        return "Subset2"
    elif "Subset3" in filename:
        return "Subset3"
    else:
        raise ValueError(f"Unknown subset in filename: {filename}")

# Move the patches to the new directory structure
for cls in classes:
    class_dir = os.path.join(source_dir, cls)
    for patch_filename in os.listdir(class_dir):
        if patch_filename.endswith(".jpg"):
            subset = determine_subset(patch_filename)
            source_path = os.path.join(class_dir, patch_filename)
            destination_dir = os.path.join(destination_base, subset, "train", cls)
            destination_path = os.path.join(destination_dir, patch_filename)
            shutil.move(source_path, destination_path)
            print(f"Moved {source_path} to {destination_path}")

print("Patches have been reorganized.")
