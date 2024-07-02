import os
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Function to get all image file paths in a directory (and subdirectories)
def get_image_files(directory):
    return [path for path in Path(directory).rglob('*.jpg')]

# Path to the patches
patch_folder = "/home/akebli/test5/patches/"

# Get the file paths for each image in the dataset
img_paths = get_image_files(patch_folder)

# Get the number of samples for each image class
class_counts = Counter(path.parent.name for path in img_paths)
# Get the class names
class_names = list(class_counts.keys())
# Print the class names
print("Class names:", class_names)
# Print the number of samples for each image class
class_counts_df = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Count'])
# Display the DataFrame
print(class_counts_df)

# Plot the class distribution
plt.figure(figsize=(10, 6))
plt.bar(class_counts_df.index, class_counts_df['Count'], color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.show()

# Save the figure
figure_save_path = "/home/akebli/test5/class_distribution.png"
plt.savefig(figure_save_path)

# Show the figure
plt.show()