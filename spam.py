# importing libraries
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

#importing datasets
file_path = 'mail_data.csv'
df = pd.read_csv(file_path)

# number of datasets
# print(len(df))
# first 5 datasets
# print(df["Category"].head())

# converting string to integer in category column

df.loc[df['Category'] == 'ham', 'Category'] = 1
df.loc[df['Category'] == 'spam', 'Category'] = 0

# splitting training and testing data
X,y = df['Message'], df['Category']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=3)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

y_test = y_test.astype('int')
y_train = y_train.astype('int')


# plt.scatter(X_train_tfidf,y_train)
# plt.show()

print(X_train_tfidf[0])

# Train a Sigmoid SVM classifier
svm_classifier = svm.SVC(kernel='sigmoidpo')
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_tfidf)


# mail = input("Enter the mail: ")
#
# input_data_features = vectorizer.transform([mail])
# pred = svm_classifier.predict(input_data_features)
# print(pred)

# if pred == 0:
#     print("SPAM!")
# else:
#     print("Not a Spam")

# Evaluate the SVM model
accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", conf_matrix)
# print("Classification Report:\n", classification_rep)


#GUI

import tkinter as tk
def perform_action():
    user_input = input_entry.get()
    input_data_features = vectorizer.transform([user_input])
    pred = svm_classifier.predict(input_data_features)
    if pred == 0:
        result_label.config(text=f"Output: SPAM !")
    if pred == 1:
        result_label.config(text=f"Output: Not a spam !")

root = tk.Tk()
root.title("Spam Detection")

prompt_label = tk.Label(root, text="Enter something:")
prompt_label.pack()

input_entry = tk.Entry(root)
input_entry.pack()

action_button = tk.Button(root, text="Submit", command=perform_action)
action_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()

