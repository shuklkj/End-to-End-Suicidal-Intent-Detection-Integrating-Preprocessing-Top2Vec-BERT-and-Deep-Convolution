# Suicidal Intent Detection - Data Preprocessing
This module offers utilities for loading and preprocessing the suicidal intent data from Twitter datasets. The main goal is to prepare the data by cleaning, standardizing, and splitting it into training, validation, and test sets.

## Features
1. Cleans tweets using a custom standardization process.
2. Option to remove stopwords during standardization.
3. Use Snowball Stemmer to stem words.
4. Supports parallel processing for efficient data cleaning.
5. Option to return test dataset and standardize it.
6. Automatic split of data into training and validation sets.

## Pre-requisites
1. Python
2. ```python
   pip install numpy
   ```
3. ```python
   pip install pandas
   ```
4. ```python
   pip install pandarallel
   ```
5. ```python
   pip install nltk
   ```
6. ```python
   pip install tweet-preprocessor
   ```
## Usage
1. Custom Standardization: Standardizes and cleans the input data.
```python
processed_data = custom_standardization(input_data)
```
2. Load Suicidal Intent Data: Load data and optionally standardize it.
```python
train_df, val_df, test_df = suicidal_intent_data_load()
```

## Functions

1. custom_standardization(input_data, remove_stopwords=True): This function processes the input tweet. It cleans, converts to lowercase, removes punctuation, and optionally removes stopwords. It also stems from the words using the Snowball Stemmer.

2. suicidal_intent_data_load(test_dataset=True, remove_stopwords=True, standardization=True): This function loads the train and test data from CSV files. It optionally standardizes the data. It then splits the train data into train and validation sets.

## Dataset Structure

The code assumes that the data is structured with columns named "target" and "text." The target column should have values of 4 for non-suicidal and 1 for suicidal intents. These are converted to 0 (non-suicidal) and 1 (suicidal) in the code.

## Troubleshooting

1. Ensure the dataset paths are correct, and the files exist in the specified location.
2. If you encounter any issues with parallel processing, try disabling pandarallel by standardizing the data without parallel processing.
3. Ensure you have all the necessary libraries installed.


