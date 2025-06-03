# 📧 Spam Email Classifier using RNN (TensorFlow & Keras)

This project builds a simple spam email classifier using a Recurrent Neural Network (RNN) with Keras and TensorFlow. It reads labeled email text data, processes it, trains a model, and allows prediction on new input emails to classify them as **Spam** or **Not Spam**.

## 📁 Dataset

The model uses a CSV file named `spam_or_not_spam.csv` with the following structure:

| Column | Description           |
|--------|-----------------------|
| email  | The email content     |
| label  | `1` for Spam, `0` for Not Spam |

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install pandas tensorflow scikit-learn
🧠 Model Architecture
Tokenizer: Converts email texts into sequences of integers.

Padding: Ensures all email inputs are of the same length (max_length = 200).

Embedding Layer: Turns word indices into dense vectors.

SimpleRNN Layer: Captures sequential patterns in the emails.

Dense Layers: Fully connected layers for binary classification.

python

Sequential([
    Embedding(vocab_size=10000, embedding_dim=64, input_length=200),
    SimpleRNN(64),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

🏃‍♂️ How to Run
Place your dataset as spam_or_not_spam.csv in the same folder.

Run the Python script.

After training, test a custom email like:

python

sample_email = "I hope this offer letter find you in good way!"
result = predict_spam(sample_email)
print("Prediction:", result)

🔮 Prediction Function
python

def predict_spam(email_text):
    sequence = tokenizer.texts_to_sequences([email_text])
    padded_seq = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded_seq)[0][0]
    return "Spam" if prediction > 0.5 else "Not Spam"
📈 Training Output
Optimizer: adam

Loss: binary_crossentropy

Metrics: accuracy

Epochs: 5 (can be increased for better performance)

Batch size: 32

✅ Example Output
Epoch 1/5
...
Accuracy: 0.95
Prediction: Not Spam
🛠️ Notes
You can improve accuracy by:

Cleaning email texts (removing HTML, punctuation, etc.)

Using more advanced models like LSTM or GRU

Training for more epochs or tuning hyperparameters
