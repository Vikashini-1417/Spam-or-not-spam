import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("spam_or_not_spam.csv")

# Tokenization and padding
vocab_size = 10000
max_length = 200
embedding_dim = 64

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df['email'].astype(str))
sequences = tokenizer.texts_to_sequences(df['email'].astype(str))
padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Labels
labels = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

# RNN model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    SimpleRNN(64),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=32)
# Function to predict if a new email is spam or not
def predict_spam(email_text):
    sequence = tokenizer.texts_to_sequences([email_text])
    padded_seq = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded_seq)[0][0]
    return "Spam" if prediction > 0.5 else "Not Spam"

# Example usage
sample_email = "I hope this offer letter find you in good way!"
result = predict_spam(sample_email)
print("Prediction:", result)
