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
import timeit
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LinearRegression
import string
from collections import Counter
from wordcloud import WordCloud
def word_freq(data, label, target, text, title):
    freq_df = data[data[target] == label]
    freq_words = freq_df[text].dropna().tolist()  # Drop NaN values
    freq_words = [str(i).lower() for i in freq_words]  # Convert to string before lowercasing
    freq_punc = []

    for o in freq_words:
        freq_punc += nltk.word_tokenize(o)

    freq_punc = [o for o in freq_punc if o not in string.punctuation]
    freq_freq = Counter(freq_punc)

    freq_top = freq_freq.most_common(100)

    words = [word for word, _ in freq_top]
    counts = [count for _, count in freq_top]

    plt.figure(figsize=(15, 25))
    plt.barh(words, counts)
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Words")

    return freq_top


def print_wordcloud(dict_top):
    dict_top = dict(dict_top)
    word_cloud = WordCloud(
        width=1200,
        height=700,
        background_color="black",
        min_font_size=5
    ).generate_from_frequencies(dict_top)

    plt.figure()
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

# Your existing code with time measurement
start_time = timeit.default_timer()

df = pd.read_csv('BG3_reviews.csv')
print(df.info())
print(df.isnull().sum())
df_cleaned = df.dropna()
print(df_cleaned.isnull().sum())

# Visualization
value_counts = df_cleaned['voted_up'].value_counts()
labels = ['Liked' if x else 'Unliked' for x in value_counts.index]
fig, ax = plt.subplots()
pie = ax.pie(value_counts, labels=labels, colors=['green', 'red'], autopct='%1.1f%%', startangle=0, counterclock=False)
counts = [f"{count} ({percentage:.1f}%)"
          for count, percentage in zip(value_counts, value_counts / value_counts.sum() * 100)]
ax.legend(pie[0], counts, title="Counts", loc="lower center", bbox_to_anchor=(0.5, -0.15))
plt.show()

UnLiked = word_freq(df, 0, "voted_up", "review", "Most Used 100 Unliked ")
Liked = word_freq(df, 1, "voted_up", "review", "Most Used 100 Liked")
print_wordcloud(UnLiked)
print_wordcloud(Liked)
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


# Function for model evaluation
# Function for model evaluation
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model_start_time = timeit.default_timer()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Convert y_pred to binary values if needed
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    report = classification_report(y_test, y_pred_binary)
    model_end_time = timeit.default_timer()
    print(f"{model_name} Model Accuracy: {accuracy:.2f}")
    print(f"Classification Report for {model_name}:\n", report)
    print(f"{model_name} Model Training and Evaluation Time: {model_end_time - model_start_time:.2f} seconds")
    print()


# Train and evaluate models
for model_name, model in models.items():
    evaluate_model(model, X_train, X_test, y_train, y_test, model_name)

# LSTM Model
max_words = 10000
max_len = 100

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
df_cleaned.loc[:, 'tokenized_text'] = df_cleaned['review'].apply(tokenizer.tokenize)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df_cleaned['review'])
X_lstm = tokenizer.texts_to_sequences(df_cleaned['tokenized_text'])
X_lstm = pad_sequences(X_lstm, maxlen=max_len)

y_lstm = df_cleaned['voted_up'].astype(int)

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

lstm_model = Sequential()
lstm_model.add(Embedding(max_words, 128, input_length=max_len))
lstm_model.add(LSTM(100))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

lstm_start_time = timeit.default_timer()
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, validation_split=0.2)
lstm_end_time = timeit.default_timer()

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
    user_input = input("Enter a review: ").strip()
    if not user_input:
        print("Invalid input. Please enter a review.")
        continue

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
    end_time = timeit.default_timer()
    print(f"Overall Processing Time: {end_time - start_time:.2f} seconds")
