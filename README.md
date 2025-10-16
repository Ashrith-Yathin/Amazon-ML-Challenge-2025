# Amazon ML Challenge 2025: Smart Product Pricing

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the solution for the Amazon ML Challenge 2025, a competition focused on predicting product prices from multimodal data (text and images). The final model is a robust ensemble of gradient boosting machines that achieved a **top-tier SMAPE score of 43.9** on the official leaderboard.

## Overview & Final Score

The challenge is to predict the price of a product given its text description (`catalog_content`) and an image (`image_link`). The relationship is complex, with factors like brand, quantity, and product quality heavily influencing the price.

This solution employs a state-of-the-art multimodal approach, combining advanced feature engineering with a powerful ensemble of models to achieve a highly competitive score.

* **Final Official SMAPE Score:** `43.9%`

## Methodology & Architecture

The core strategy is to create a rich, high-dimensional feature set from the raw data and then train a diverse set of powerful models to make a final, blended prediction.

### 1. Feature Engineering
The raw text and image data are converted into numerical "embeddings" and combined with manually extracted features.
* **Text Features:**
    * **Semantic Embeddings:** A `SentenceTransformer` (`all-MiniLM-L6-v2`) is used to convert the cleaned text into 384-dimensional vectors that capture its meaning.
    * **"Golden Features":** Critical information like **Brand**, **Quantity**, and **Unit** are explicitly extracted from the `catalog_content` using regex and custom logic.

* **Image Features:**
    * **Visual Embeddings:** A pre-trained Vision Transformer from OpenAI's CLIP library (`ViT-B/16`) is used to convert each product image into a 512-dimensional vector, capturing visual cues about quality, category, and brand.

* **Final Feature Set:** All features (text embeddings, image embeddings, quantity, and one-hot encoded brand/unit) are concatenated into a single, wide feature array for each product.

### 2. Modeling Strategy: K-Fold Ensemble
To ensure robustness and maximize accuracy, a **5-Fold Cross-Validation** strategy is used to train an ensemble of three powerful gradient boosting models:
* **LightGBM:** Known for its speed and efficiency.
* **XGBoost:** A highly robust and accurate model, accelerated on the Mac's M1 GPU (`mps`).
* **CatBoost:** Another powerful model, excellent at handling categorical nuances.

The final prediction is a weighted average of the predictions from all 15 models (3 model types x 5 folds).

## Setup and Installation

Follow these steps to set up the project environment on your local machine.

### Prerequisites
* Python 3.9+
* `venv` module for virtual environments

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Ashrith-Yathin/Amazon-ML-Challenge-2025.git](https://github.com/Ashrith-Yathin/Amazon-ML-Challenge-2025.git)
    cd Amazon-ML-Challenge-2025
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    The project includes a cell in `main.ipynb` to install all dependencies. Run the first cell in the notebook to install everything.
    ```python
    # Run this in your notebook
    %pip install -U sentence-transformers timm torch torchvision pandas numpy scikit-learn tqdm psutil
    %pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)
    ```

## How to Run the Pipeline

The entire pipeline is contained within the `workflow.ipynb` Notebook. The steps are designed to be run sequentially.

1.  **Download Data:** Untar the `train.csv` and `test.csv` zip files. The notebook contains cells to download all the necessary product images into the `images/` folder.

2.  **Generate Features:** Run the cells in the notebook to:
    * Parse the `catalog_content` and extract brand/quantity features.
    * Generate text embeddings and save them to the `embeddings/` folder.
    * Generate the upgraded image embeddings (`ViT-B/16`) and save them to the `embeddings_medium/` folder.
    * Combine all features into the final `final_X_..._with_brand.npy` files.

3.  **Train the Final Model:** Run the final cell, which performs memory-safe K-Fold training on the ensemble of LightGBM, XGBoost, and CatBoost.

4.  **Generate Submission:** The script will automatically create the `test_out.csv` file in the root directory, ready for upload.
