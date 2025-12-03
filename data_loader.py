import os
import kagglehub

DATA_DIR = "datasets"

def download_data():
    """Download dataset from KaggleHub."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia", path=DATA_DIR)
    print("Dataset downloaded to:", path)

if __name__ == "__main__":
    download_data()
