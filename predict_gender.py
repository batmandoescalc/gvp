from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
'''
X = list of story texts 
y = pronoun labels ("she/her", "he/him")
//each element in x is a full story corresponding to the pronouns in y.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1,2),
        min_df=3
    )),
    ("clf", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)

preds = model.predict(X_test)

print(classification_report(y_test, preds))
'''