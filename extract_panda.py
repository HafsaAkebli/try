import zipfile
import shutil
import os

# Define the paths
zip_file_path = '/home/akebli/prostate-cancer-grade-assessment.zip'
extract_path = '/home/akebli/panda_dataset'
copy_path = '/mnt/dmif-nas/MITEL/challenges/panda_dataset'

# Create the extraction directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Extract the zip file
#with zipfile.ZipFile(zip_file_path, 'r') as z:
 #   z.extractall(path=extract_path)
#print("Extraction and copying completed")

# Copy the extracted folder to the new location
shutil.copytree(extract_path, copy_path)

print(f"Extraction and copying completed. Files are now in {copy_path}")
