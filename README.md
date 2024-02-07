# Text Classification with Machine Learning and BERT

Welcome to our GitHub repository dedicated to text classification leveraging the power of both traditional machine learning models and the cutting-edge BERT (Bidirectional Encoder Representations from Transformers) language model. This project encompasses a wide range of techniques, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation, all aimed at providing a comprehensive understanding and implementation of text classification tasks.

## Prerequisites

Ensure you have the following Python libraries installed to run the code smoothly:

- `pandas`
- `nltk`
- `re`
- `seaborn`
- `matplotlib`
- `sklearn`
- `wordcloud`
- `xgboost`
- `torch`
- `transformers` (for BERT)

You can easily install these dependencies with the following pip command:


## Usage

To get started, clone this repository to your local machine or download the code files directly. Ensure you have your dataset in CSV format and specify the file path in the `file_path` variable within the code. You can run the code step-by-step or execute the entire script as per your preference.

## Contents

- **Data Loading and Exploration:** Initial steps to load the dataset, visualize its structure, and understand the data through summary statistics and category distributions. It also includes checks for missing values.
- **Data Preprocessing:** This section covers cleaning the data by removing duplicates, tokenization, converting to lowercase, filtering non-alphabetic tokens and stopwords, and applying optional stemming and lemmatization.
- **Exploratory Data Analysis (EDA):** We dive deeper into the data with visualizations such as word clouds, frequency distributions of top N words, text length distributions, and category-wise box plots for text length.
- **Data Splitting:** The dataset is divided into training (70%), validation (15%), and testing (15%) sets.
- **Feature Engineering:** Transformation of text data into numerical features using TF-IDF vectorization and label encoding for target labels.
- **Model Training:** Training of various machine learning models including Random Forest, Logistic Regression, Support Vector Machine, Multinomial Naive Bayes, K-Nearest Neighbors, Gradient Boosting, XGBoost, and a Neural Network.
- **Model Evaluation:** Evaluation of models based on accuracy and F1 score, including the generation of a confusion matrix for the best-performing model.
- **BERT-based Language Model:** Implementation and fine-tuning of a BERT-based language model for text classification, including tokenization and encoding of the data.

## References

For further reading and documentation on the tools and libraries used in this project:

- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Transformers (Hugging Face)](https://huggingface.co/transformers/)
- [BERT Base German Cased](https://huggingface.co/bert-base-german-cased)

We highly recommend referring to the code comments for detailed explanations of each step involved. Enjoy exploring text classification with the combined strength of machine learning and BERT!

