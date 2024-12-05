# Toxic Comment Classification
## Overview
This project builds a machine learning pipeline to classify toxic comments from Wikipedia into six categories:

1.Toxic  
2.Severe Toxic   
3.Obscene   
4.Threat  
5.Insult   
6.Identity Hate   
The goal is to predict the probabilities of each toxicity type for a given comment using machine learning models, optimizing both accuracy and AUC scores.   

## Dataset
Source: Wikipedia comments dataset.  
Size: 159,571 labeled comments.
## Features:
Text comments with toxicity labels for six categories.  
Labels: Binary (0 for non-toxic, 1 for toxic).
## Preprocessing:
Removed punctuations, hyperlinks, numbers, and non-alphanumeric characters.  
Performed lemmatization using WordNetLemmatizer from nltk.  
Vectorized comments using TfidfVectorizer with a maximum of 5,000 features.  
Models Implemented
1. Logistic Regression
Strengths: Easy to interpret, performs well for linear relationships.  
Weaknesses: Limited to linear decision boundaries.  
### **Metrics**

| Toxicity Type   | Accuracy (%) | AUC Score |
|-----------------|--------------|-----------|
| Toxic           | 90           | 0.4981    |
| Severe Toxic    | 99           | 0.5276    |
| Obscene         | 95           | 0.5456    |
| Threat          | 100          | 0.4759    |
| Insult          | 95           | 0.4912    |
| Identity Hate   | 99           | 0.4779    |

2. Multinomial Naive Bayes
Strengths: Fast and efficient for text classification tasks.  
Weaknesses: Relies on the assumption of feature independence.
### **Metrics**

| Toxicity Type   | Accuracy (%) | AUC Score |
|-----------------|--------------|-----------|
| Toxic           | 37           | 0.4186    |
| Severe Toxic    | 65           | 0.3415    |
| Obscene         | 42           | 0.4104    |
| Threat          | 80           | 0.3840    |
| Insult          | 42           | 0.4019    |
| Identity Hate   | 67           | 0.3972    |

3. Random Forest
Strengths: Robust and less prone to overfitting.  
Weaknesses: Computationally expensive for large datasets.  
### **Metrics**

| Toxicity Type   | Accuracy (%) | AUC Score |
|-----------------|--------------|-----------|
| Toxic           | 89           | 0.5190    |
| Severe Toxic    | 99           | 0.5463    |
| Obscene         | 94           | 0.5665    |
| Threat          | 100          | 0.5050    |
| Insult          | 95           | 0.5089    |
| Identity Hate   | 99           | 0.5425    |

## Methodology
### Exploratory Data Analysis:
Visualized class distributions and word frequencies using word clouds and bar plots.
### Preprocessing:
Removed stopwords, unnecessary characters, and lemmatized words.  
Vectorized text data using TfidfVectorizer.
### Model Training:
Trained Logistic Regression, Multinomial Naive Bayes, and Random Forest classifiers.
### Evaluation:
Assessed models using accuracy, AUC, and ROC curves.
### Key Findings
Logistic Regression and Random Forest performed well in terms of accuracy.  
Random Forest achieved the highest AUC scores across most toxicity types, making it the best overall model.
### Technologies Used
Programming Language: Python
### Libraries:
Text Preprocessing: nltk, pandas, TfidfVectorizer from sklearn
Machine Learning: scikit-learn, matplotlib
Data Visualization: Seaborn, matplotlib
