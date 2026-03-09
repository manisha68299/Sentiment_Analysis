# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1️⃣ Load Dataset
# ------------------------------------------------
df = pd.read_csv("imdb_reviews_data.csv")

print("🔎 First 5 rows of dataset:")
print(df.head(), "\n")

print("📊 Class distribution:")
df = df.rename(columns={"Review": "review", "level": "label"})


# ------------------------------------------------
# 2️⃣ Preprocessing
# ------------------------------------------------
# Lowercase and strip extra spaces
df['review'] = df['review'].str.lower().str.strip()

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(df['label'])

# Prepare string target names for reporting
target_names = [str(c) for c in le.inverse_transform(range(len(le.classes_)))]

# ------------------------------------------------
# 3️⃣ Train/Test Split
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ------------------------------------------------
# 4️⃣ Build Pipeline (TF-IDF + Logistic Regression)
# ------------------------------------------------
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# ------------------------------------------------
# 5️⃣ Train Model
# ------------------------------------------------
model.fit(X_train, y_train)

# ------------------------------------------------
# 6️⃣ Evaluate Model
# ------------------------------------------------
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📑 Classification Report:\n", 
      classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, 
                                      display_labels=target_names, cmap="Greens")
plt.title("Confusion Matrix - IMDB Sentiment Analysis")
plt.show()

# ------------------------------------------------
# 7️⃣ Prediction Functions
# ------------------------------------------------
def predict_sentiment(texts):
    """
    Predict sentiment label(s) for single or multiple texts.
    Returns human-readable labels as strings.
    """
    preds = model.predict(texts)
    return le.inverse_transform(preds)

def predict_sentiment_proba(texts):
    """
    Return sentiment probabilities for single or multiple texts.
    Returns a list of dictionaries with probabilities for each class.
    """
    proba = model.predict_proba(texts)
    return [{cls: round(p, 3) for cls, p in zip(target_names, prob)} for prob in proba]

# ------------------------------------------------
# 8️⃣ Example Usage
# ------------------------------------------------
examples = [
    "The movie was amazing and the acting was fantastic!",
    "It was a boring movie, too long and dull.",
    "Average movie, not bad but not great either."
]

for ex in examples:
    print(f"\nReview: '{ex}'")
    print("Predicted Sentiment:", predict_sentiment([ex])[0])
    print("Probabilities:", predict_sentiment_proba([ex])[0])
