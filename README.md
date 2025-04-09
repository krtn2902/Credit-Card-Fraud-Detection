# Credit Card Fraud Detection Project

This project aims to build a machine learning model to detect fraudulent credit card transactions. It uses a Random Forest classifier trained on a dataset of credit card transactions.

## Project Structure

```
Directory structure:
    ├── README.md
    ├── requirements.txt
    ├── models/
    │   └── random_forest_model.pkl
    ├── notebooks/
    │   ├── 01_data_preprocessing.ipynb
    │   └── 02_model_training.ipynb
    └── outputs/
        └── evaluation_report.txt
```
## Workflow

1.  **Data Preprocessing (`notebooks/01_data_preprocessing.ipynb`):**
    *   Loads the raw dataset (`data/creditcard.csv`).
    *   Performs basic data exploration (info, description, null checks).
    *   Visualizes class distribution and correlations.
    *   Applies StandardScaler to 'Amount' and 'Time' features.
    *   Drops original 'Time' and 'Amount' columns.
    *   Saves the processed data to `data/processed_data.csv`.

2.  **Model Training (`notebooks/02_model_training.ipynb`):**
    *   Loads the processed data (`data/processed_data.csv`).
    *   Splits the data into training and testing sets.
    *   Trains a `RandomForestClassifier` model.
    *   Evaluates the model using classification report, confusion matrix, and ROC AUC score.
    *   Saves the trained model to `models/random_forest_model.pkl`.
    *   Saves the evaluation results to `outputs/evaluation_report.txt`.
    *   Generates and saves the ROC curve plot to `outputs/roc_curve.png`.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
   * Linux/MacOS
   ```bash
    python -m venv venv
    source venv/bin/activate
   ```
   * Windows
   ```bash
    python -m venv venv
    venv\Scripts\activate
   ```
4.  **Install dependencies:**
    *   Ensure the `requirements.txt` file lists all necessary packages. Based on the notebooks, you'll likely need: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
    *   Install using pip:
        ```bash
        pip install -r requirements.txt
        ```
5.  **Add Data:**
    *   Place the raw dataset file (`creditcard.csv`) into the `data/` directory. (Note: This dataset is often found on Kaggle).

## How to Run

1.  Ensure the `creditcard.csv` file is in the `data/` directory.
2.  Run the data preprocessing notebook:
    *   Execute the cells in `notebooks/01_data_preprocessing.ipynb`. This will generate `data/processed_data.csv`.
3.  Run the model training notebook:
    *   Execute the cells in `notebooks/02_model_training.ipynb`. This will train the model, save it to `models/`, and generate evaluation files in `outputs/`.

## Results

The model performance metrics (precision, recall, F1-score, confusion matrix, AUC score) are available in `outputs/evaluation_report.txt`. The ROC curve visualization is saved as `outputs/roc_curve.png`
