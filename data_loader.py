import os
import logging
import kagglehub

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the directory where datasets will be stored
DATA_DIR = "datasets"

def download_data(dataset_name):
    """Download dataset from KaggleHub."""
    try:
        # Check if the dataset directory exists, if not, create it
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            logger.info(f"Created directory: {DATA_DIR}")

        # Download the dataset
        path = kagglehub.dataset_download(dataset_name, path=DATA_DIR)
        logger.info(f"Dataset downloaded to: {path}")
        return path
    except kagglehub.exceptions.KaggleApiHTTPError as e:
        logger.error(f"Error downloading dataset: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    # Define the dataset to download
    dataset_name = "paultimothymooney/chest-xray-pneumonia"
    
    # Download the dataset
    downloaded_path = download_data(dataset_name)
    
    if downloaded_path:
        logger.info(f"Dataset successfully downloaded to: {downloaded_path}")
    else:
        logger.error("Failed to download dataset.")
