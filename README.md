# Text Classification with Machine Learning and BERT

This GitHub repository contains Python code for performing text classification tasks using a combination of machine learning models and a BERT-based language model. The code includes data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation.

## Prerequisites
Before running the code, make sure you have the following libraries installed:

- pandas
- nltk
- re
- seaborn
- matplotlib
- sklearn
- wordcloud
- xgboost
- torch
- transformers (for BERT)

You can install these libraries using pip:

```bash
pip install pandas nltk seaborn matplotlib scikit-learn wordcloud xgboost torch transformers


Usage
Clone this repository to your local machine or download the code file.

Make sure you have the required dataset in CSV format. You can specify the file path in the file_path variable.

Run the code step by step or execute the entire script.

Contents
Data Loading and Exploration: The code loads the dataset, displays its structure, summary statistics, category distribution, and checks for missing values.

Data Preprocessing: It includes removing duplicates, tokenization, lowercasing, removing non-alphabetic tokens and stopwords, and optional stemming and lemmatization.

Exploratory Data Analysis (EDA): Visualizations are created to explore the data, including word clouds, top N common words, text length distribution, and box plots for text length by category.

Data Splitting: The data is split into training (70%), validation (15%), and test (15%) sets.

Feature Engineering: Text data is transformed into numerical features using TF-IDF vectorization. Label encoding is performed on the target labels.

Model Training: Several machine learning models are trained, including Random Forest, Logistic Regression, Support Vector Machine, Multinomial Naive Bayes, K-Nearest Neighbors, Gradient Boosting, XGBoost, and a Neural Network.

Model Evaluation: The models are evaluated using accuracy and F1 score. A confusion matrix is generated for the best-performing model (Random Forest).

BERT-based Language Model: A BERT-based language model is loaded and fine-tuned for text classification. Tokenization and encoding are performed on the data. The model is trained and evaluated, and a confusion matrix is generated.

References
scikit-learn
XGBoost
Transformers (Hugging Face)
BERT Base German Cased
Please refer to the code comments for detailed explanations of each step. Enjoy text classification with machine learning and BERT!
