# Email and SMS Spam Classifier

This project aims to classify emails and SMS messages as spam or not spam using machine learning techniques.

## Features

- Train a spam classifier using a labeled dataset.
- Evaluate the performance of the model.
- Deploy the model as a web application for real-time spam detection.

## Deployment

The app is deployed and can be accessed [here](https://sms-spam-classifier-by-phani.streamlit.app/).


## Files

- `app.py`: Streamlit application for running the classifier.
- `requirements.txt`: Lists required Python packages.
- `sms_spam_detection.ipynb`: Jupyter Notebook for training and evaluating the model and for data understanding using Exploratory Data Analysis.
- `spam.csv`: Dataset containing labeled messages.
- `model_files/`: Directory containing vectorizer and trained model files.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Phani943/Email_Sms_Spam_Classifier.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Email_Sms_Spam_Classifier
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Training the Model**:
    - Open and run the `sms_spam_detection.ipynb` notebook to train and evaluate the spam classifier.

2. **Running the Web Application**:
    - Start the Streamlit app:
        ```sh
        streamlit run app.py
        ```
    - Open your web browser and go to url provided there, to use the spam classifier.

## Dataset

The `spam.csv` file contains the labeled messages used for training. Each message is labeled as either "spam" or "ham" (not spam).

## Acknowledgements

- The dataset used in this project is sourced from the
- [Kaggle/UCI Machine Learning](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
