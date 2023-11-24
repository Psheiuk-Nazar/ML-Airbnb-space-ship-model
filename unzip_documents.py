import os
import zipfile

archive_path = "C:\\Users\\Owner\\Documents\\Portfolio\\Ships\\airbus-ship-detection.zip"

extracted_path = "C:\\Users\\Owner\\Documents\\Portfolio\\Ships"

os.makedirs(extracted_path, exist_ok=True)

with zipfile.ZipFile(archive_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)
