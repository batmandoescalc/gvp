from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

X = #list of story texts
y = #pronoun labels ("she/her", "he/him")
#each element in x is a full story corresponding to the pronouns in y.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)
#set it up so that all of one authors documents are all in one set either train or test, stratified sampling
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="Fill with chosen stop words", #stop words needs to be filled with what we choose
        max_features=20000,
        ngram_range=(1,2), #pay attention to single words and two word phrases
        min_df=5 #remove words that appear in fewer than 3 documents
        #remove unique words to avoid overfit 
        
    )),
    ("clf", LogisticRegression(max_iter=2000, penalty = "l1", solver="liblinear", C = 1.0))
])
#lasso regression technique

model.fit(X_train, y_train)
preds = model.predict(X_test)

print(classification_report(y_test, preds))

'''
Talk with matt: 
look up visualisatizon of this model 
minimum want some kind of feature importance to see which words are most indicative
validate the model, this should help figure out what we want to do with the data/what words and phrases we want to keep track of

After matt:
Things to consider: 
if the texts have pronouns in them, then the model may simply memorize pronoun tokens instead of learning writing patterns/themes.
We need to also consider the balance of texts, we have a lot for "she/her" and for "he/him", but not for "they/them" and "aer/aers" so the model may be biased towards predicting "she/her" and "he/him".
Version A — Keep pronouns
Measures:
“Can the model directly identify pronoun-associated language?”

Version B — Remove pronouns before training
Measures:
“Are there broader linguistic patterns associated with pronoun groups?”

do both and compare results, see if the model is just picking up on pronouns or if it can learn other patterns.
to do After training the model:  
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

vectorizer = model.named_steps["tfidf"]
clf = model.named_steps["clf"]

feature_names = np.array(vectorizer.get_feature_names_out())

for i, class_label in enumerate(clf.classes_):
    top20 = np.argsort(clf.coef_[i])[-20:][::-1]  # sorted strongest → weakest

    print(f"\nTop words for {class_label}:")
    print(feature_names[top20])

#other visualization option: 
target_class = "they/them" #whichever class we want to analyze, just put they/them
class_index = list(clf.classes_).index(target_class)

top = np.argsort(clf.coef_[class_index])[-15:][::-1]

words = feature_names[top]
weights = clf.coef_[class_index][top]

plt.figure(figsize=(10, 5))
plt.barh(words, weights)

plt.xlabel("Coefficient Weight")
plt.ylabel("Features")
plt.title(f"Top predictive features for {target_class}")

plt.gca().invert_yaxis()  # makes strongest feature appear at top
plt.show()

#confusion matrix for validation:
ConfusionMatrixDisplay.from_predictions(
    y_test,
    preds,
    display_labels=clf.classes_,
    xticks_rotation=45
)

plt.show()

#checking pronoun distribution in the dataset, we have these numbers in the data Adam went through: 
from collections import Counter
print(Counter(y))