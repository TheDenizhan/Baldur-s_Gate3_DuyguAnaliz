import pandas as pd
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from xgboost import XGBClassifier
import time
from sklearn.linear_model import LinearRegression

def ds_info(df, ds_name='df'):
    print(f"The {ds_name} dataset info:\n")
    print(df.info())

def check_null(df, ds_name='df'):
    print(f"Null Values in each col in the {ds_name} dataset:\n")
    print(df.isnull().sum())

def drop_null_rows(df, ds_name='df'):
    print(f"Dropping null rows in the {ds_name} dataset...\n")
    df_cleaned = df.dropna()
    print(f"New shape after dropping null rows: {df_cleaned.shape}")
    return df_cleaned

df = pd.read_csv('BG3_reviews.csv')
ds_info(df)
check_null(df)
df_cleaned = drop_null_rows(df)
check_null(df_cleaned)

# Your existing code with time measurement
start_time = time.time()

value_counts = df_cleaned['voted_up'].value_counts()
labels = ['Liked' if x else 'Unliked' for x in value_counts.index]
fig, ax = plt.subplots()
pie = ax.pie(value_counts, labels=labels, colors=['green', 'red'], autopct='%1.1f%%', startangle=0, counterclock=False)
counts = [f"{count} ({percentage:.1f}%)"
          for count, percentage in zip(value_counts, value_counts / value_counts.sum() * 100)]

ax.legend(pie[0], counts, title="Counts", loc="lower center", bbox_to_anchor=(0.5, -0.15))
plt.show()

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf_vectorizer.fit_transform(df_cleaned['review'])
y = df_cleaned['voted_up']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': make_pipeline(ColumnTransformer(
        transformers=[
            ('num', StandardScaler(with_mean=False), slice(0, -1))  # Use with_mean=False for sparse matrices
        ],
        remainder='passthrough'
    ), KNeighborsClassifier(n_neighbors=5)),
    'SVM': SVC(),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models
for model_name, model in models.items():
    model_start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"{model_name} Model Accuracy: {accuracy:.2f}")
    print(f"Classification Report for {model_name}:\n", report)
    model_end_time = time.time()
    print(f"{model_name} Model Training and Evaluation Time: {model_end_time - model_start_time:.2f} seconds")
    print()

# LSTM Model
max_words = 5000
max_len = 100

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
df_cleaned.loc[:, 'tokenized_text'] = df_cleaned['review'].apply(tokenizer.tokenize)


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df_cleaned['review'])
X_lstm = tokenizer.texts_to_sequences(df_cleaned['review'])
X_lstm = pad_sequences(X_lstm, maxlen=max_len)

y_lstm = df_cleaned['voted_up'].astype(int)

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

lstm_model = Sequential()
lstm_model.add(Embedding(max_words, 128, input_length=max_len))
lstm_model.add(LSTM(100))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

lstm_start_time = time.time()
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, validation_split=0.2)
lstm_end_time = time.time()

# Evaluate LSTM model
y_pred_lstm_prob = lstm_model.predict(X_test_lstm)
y_pred_lstm = (y_pred_lstm_prob > 0.5).astype(int)
accuracy_lstm = accuracy_score(y_test_lstm, y_pred_lstm)
report_lstm = classification_report(y_test_lstm, y_pred_lstm)
print(f"LSTM Model Accuracy: {accuracy_lstm:.2f}")
print(f"Classification Report for LSTM:\n", report_lstm)
print()

# User input prediction
while True:
    user_input = input("Enter a review: ")
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    user_input_lstm = pad_sequences(tokenizer.texts_to_sequences([user_input]), maxlen=max_len)
    for model_name, model in models.items():
        prediction = model.predict(user_input_tfidf)
        print(f"{model_name} Predicted Result: {'Liked' if prediction[0] else 'Unliked'}")

    # LSTM prediction
    prediction_lstm_prob = lstm_model.predict(user_input_lstm)
    prediction_lstm = (prediction_lstm_prob > 0.5).astype(int)
    print(f"LSTM Predicted Result: {'Liked' if prediction_lstm[0][0] == 1 else 'Unliked'}")

    # Overall processing time
    end_time = time.time()
    print(f"Overall Processing Time: {end_time - start_time:.2f} seconds")
