# Email Spam Detector

A machine learning application that classifies emails as spam or legitimate (ham) using natural language processing and logistic regression.

## Overview

This email spam detector uses a supervised machine learning approach to identify unwanted spam emails. The system is trained on a labeled dataset of emails and can process multiple emails to predict whether they are spam or legitimate communications.

## Features

- Text preprocessing and feature extraction using TF-IDF vectorization
- Machine learning classification using Logistic Regression
- Batch processing capability to classify multiple emails at once
- High accuracy in distinguishing between spam and legitimate emails

## Requirements

- Python 3.6+
- Required packages:
  - numpy
  - pandas
  - scikit-learn

## Installation

1. Clone this repository or download the source code.
2. Install the required packages:

```bash
pip install numpy pandas scikit-learn
```

3. Ensure you have the 'mail_data.csv' dataset file in the same directory as the script.

## Dataset

The system uses a dataset ('mail_data.csv') that contains labeled emails with:
- Category: "spam" or "ham" (legitimate)
- Message: The email content

## Usage

1. Run the script:

```bash
python spam_detector.py
```

2. The program will:
   - Load and preprocess the email data
   - Train a logistic regression model on the dataset
   - Display accuracy metrics for the training and test sets
   - Process all emails in the dataset and predict their classification
   - Display the results with a preview of each email and its prediction

## How It Works

1. **Data Preprocessing**:
   - The system loads labeled email data
   - Replaces any null values with empty strings
   - Converts "spam" and "ham" labels to 0 and 1 respectively

2. **Feature Extraction**:
   - Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
   - Converts text data into numerical feature vectors
   - Applies preprocessing like removing English stop words and converting to lowercase

3. **Model Training**:
   - Splits data into training (80%) and testing (20%) sets
   - Trains a Logistic Regression classifier on the training data
   - Evaluates accuracy on both training and test sets

4. **Prediction**:
   - Processes all emails in the dataset
   - Makes predictions based on the trained model
   - Displays each email with its classification (SPAM or HAM)

## Customization

You can modify the script to:
- Use a different dataset by changing the file name in `pd.read_csv()`
- Adjust the test/train split ratio by modifying the `test_size` parameter
- Process emails from a different source by modifying the prediction section

## Future Improvements

- Add a web or GUI interface
- Implement additional classification algorithms for comparison
- Add capability to process email attachments and headers
- Improve feature extraction with more advanced NLP techniques

## License

This project is available under the MIT License.

## Author

Nikhil Khatri (nikk-hhil)

---

Feel free to contribute to this project by submitting issues or pull requests!