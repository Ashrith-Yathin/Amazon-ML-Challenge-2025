# In src/utils.py

import os
import ssl
import urllib.request
from pathlib import Path
from functools import partial
from tqdm import tqdm
import multiprocessing

# --- This is the worker function for a single download ---
# It's defined at the top level of the file so new processes can find it.
def download_image_worker(task):
    """
    Downloads a single image.
    Accepts a tuple: (sample_id, image_link, savefolder)
    """
    sample_id, image_link, savefolder = task
    
    # Use the sample_id for the filename, which is more reliable and consistent.
    image_save_path = os.path.join(savefolder, f"{sample_id}.jpg")

    if not os.path.exists(image_save_path):
        if not isinstance(image_link, str):
            return  # Skip if the link is not a valid string
        try:
            # THE SSL FIX: This bypasses certificate verification errors.
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(image_link, image_save_path)
        except Exception:
            # Silently ignore downloads that fail (e.g., dead links).
            pass
    return

def download_images(image_tasks, download_folder):
    """
    Downloads a list of image tasks in parallel using multiprocessing.
    'image_tasks' should be a list of tuples: [(sample_id, image_link), ...]
    """
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Add the download_folder to each task to create the full task list for the workers.
    full_tasks = [(sid, link, download_folder) for sid, link in image_tasks]
    
    # Use a reasonable number of processes for a local machine.
    num_processes = max(1, os.cpu_count() - 1) 

    # Use the Pool to run the worker function on all tasks in parallel.
    with multiprocessing.Pool(num_processes) as pool:
        list(tqdm(pool.imap_unordered(download_image_worker, full_tasks), total=len(full_tasks), desc=f"Downloading to {Path(download_folder).name}"))