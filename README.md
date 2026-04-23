# IMDB Sentiment Analysis

## Problem Statement
Classify movie reviews as positive or negative using machine learning.

---

## Features
- Text preprocessing (lowercase, strip spaces)
- TF-IDF vectorization (unigrams + bigrams)
- Logistic Regression model
- Train-test split (80/20)
- Accuracy and classification report
- Confusion matrix visualization
- Prediction function for new reviews
- Probability prediction for each class

---

## Tech Stack
- Python  
- pandas  
- scikit-learn  
- matplotlib  

---

## Project Structure
## 📁 Project Structure

```
Sentiment_Analysis/
├── imdb_reviews_data.csv        # Dataset
├── SentimentAnalysis_IMDB.py    # Main script
└── README.md                    # Project documentation
```
---

## Workflow
1. Load dataset (CSV)
2. Preprocess text (lowercase, clean)
3. Encode labels
4. Split data (train/test)
5. Apply TF-IDF vectorization
6. Train Logistic Regression model
7. Evaluate using accuracy and classification report
8. Visualize confusion matrix
9. Predict sentiment for new inputs

---

## Output
- Accuracy score  
- Classification report  
- Confusion matrix plot  

---

## Example
Input: "The movie was amazing and the acting was fantastic!"  
Output: 1 (Positive)

Input: "It was a boring movie, too long and dull."  
Output: 0 (Negative)

---

## How to Run
```bash
pip install pandas scikit-learn matplotlib
python SentimentAnalysis_IMDB.py