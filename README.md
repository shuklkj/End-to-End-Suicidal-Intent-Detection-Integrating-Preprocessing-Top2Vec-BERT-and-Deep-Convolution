## Suicidal Intent Detection - Data Preprocessing
This module offers utilities for loading and preprocessing the suicidal intent data from Twitter datasets. The main goal is to prepare the data by cleaning, standardizing, and splitting it into training, validation, and test sets.

### Features
1. Cleans tweets using a custom standardization process.
2. Option to remove stopwords during standardization.
3. Use Snowball Stemmer to stem words.
4. Supports parallel processing for efficient data cleaning.
5. Option to return test dataset and standardize it.
6. Automatic split of data into training and validation sets.

### Pre-requisites
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
### Usage
1. Custom Standardization: Standardizes and cleans the input data.
```python
processed_data = custom_standardization(input_data)
```
2. Load Suicidal Intent Data: Load data and optionally standardize it.
```python
train_df, val_df, test_df = suicidal_intent_data_load()
```

### Functions

1. custom_standardization(input_data, remove_stopwords=True): This function processes the input tweet. It cleans, converts to lowercase, removes punctuation, and optionally removes stopwords. It also stems from the words using the Snowball Stemmer.

2. suicidal_intent_data_load(test_dataset=True, remove_stopwords=True, standardization=True): This function loads the train and test data from CSV files. It optionally standardizes the data. It then splits the train data into train and validation sets.

### Dataset Structure

The code assumes that the data is structured with columns named "target" and "text." The target column should have values of 4 for non-suicidal and 1 for suicidal intents. These are converted to 0 (non-suicidal) and 1 (suicidal) in the code.

### Troubleshooting

1. Ensure the dataset paths are correct, and the files exist in the specified location.
2. If you encounter any issues with parallel processing, try disabling pandarallel by standardizing the data without parallel processing.
3. Ensure you have all the necessary libraries installed.






## Twitter Text Analysis with Top2Vec
Analyze and visualize topic modeling on Twitter data using Top2Vec. This script preprocesses the tweets, extracts relevant topics, and can generate word clouds for specific topics.

### Features
1. Data preprocessing using regular expressions, NLTK, and tweet preprocessor.
2. Tokenization and cleaning of tweets.
3. Topic modeling with Top2Vec.
4. Generation of word clouds for visualizing topics.
   
### Pre-requisites
Python
1. ```python
   pip install tqdm
   ```
2. ```python
   pip install Top2Vec
   ```

### Setup
1. Clone this repository.
2. Navigate to the repository's directory.
3. Install the necessary libraries using pip.
4. Ensure the dataset is placed in the appropriate directory.
   
### How to Use
1. Run the script.
2. The processed dataset will be filtered to only contain tweets with the class_label of 1.
3. The tweets are tokenized and cleaned.
4. Top2Vec is used to model topics from the tweets.
5. Save the Top2Vec model for later use.
6. Load the saved model and generate word clouds for visualizing the topics.

### Functions Explained
1. custom_standardization(input_data): Cleans the tweet using the tweet preprocessor and converts it to lowercase.
2. tokenize_and_clean(text): Tokenizes the tweet, removes noise (such as single-character tokens, numbers, and punctuation), and removes stopwords.
   
### Customization
1. Modify the dataset path if required.
2. Adjust the number in model.generate_topic_wordcloud(17) to visualize a different topic's word cloud.
   
## Troubleshooting
1. If you encounter issues with specific libraries, ensure they are installed correctly.
2. Ensure the dataset is correctly formatted and placed in the right directory.




## Depression Detection using BERT

This repository provides code to train and evaluate a classifier for depression detection based on tweets. We fine-tune the RoBERTa model on our dataset, with specific preprocessing steps. Here's an outline:

### Overview:

1. Data Preparation: Read the dataset, preprocess the data and split it into training and validation sets.
2. Model: A custom BERTClassifier is used, which builds upon the RoBERTa model and adds additional dense layers for classification.
3. Training: Train the model on the training dataset.
4. Evaluation: Evaluate the trained model on the test dataset to get accuracy results.

### Steps:

#### 1. Data Preparation:
1. The dataset is read using pandas and basic preprocessing is performed.
2. The target variable is mapped to binary values (0 and 1).
3. The data is split into training and validation sets using train_test_split from sklearn.

#### 2. Preprocessing:
1. Tweets are cleaned using the preprocessor library to remove URLs, mentions, etc., and then converted to lowercase.
2. The dataset is tokenized using the tokenizer from the transformers library for the RoBERTa model.

#### 3. Model Architecture:
1. A custom BertClassifier class is defined which uses the pretrained RoBERTa model and additional linear layers.
2. Activation functions and dropout layers are also added.

#### 4. Training:
1. The model is trained on the training dataset using the Adam optimizer and CrossEntropyLoss as the criterion.
2. Training and validation losses, as well as accuracies, are printed for each epoch.

#### 5. Evaluation:
A separate test dataset is loaded, preprocessed, and then used to evaluate the model's accuracy.

### Usage:

#### Prerequisites:

Ensure you have the following libraries installed:
1. pandas
2. numpy
3. torch
4. transformers
5. preprocessor
6. tqdm
7. pandarallel
   
#### Running the Code:

1. Load your dataset in the specified format. Ensure that the dataset has a 'target' column with values to indicate sentiment and a 'text' column with the actual tweets.
2. Run the code. The model will be trained on the training set and evaluated on the test set.

#### Results:
The final test accuracy will be printed at the end of the script.




























