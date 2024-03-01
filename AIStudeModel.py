# %%
import pandas as pd

# %%
#Read CSV Train Data
train = pd.read_csv('C:/Users/megha/Desktop/Capstone/train_essays.csv')
train.head()

# %%
#Check for / remove missing values
train.info()

# %%
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import seaborn as sns

import matplotlib.pyplot as plt


# %%
#Need to get an idea of sequence length in the text as a hyperparamter for the model

# Tokenize text and count number of words
train['word_count'] = train['text'].apply(lambda x: len(word_tokenize(x)))

# Calculate average, max, and min number of words
average_word_count = train['word_count'].mean()
max_word_count = train['word_count'].max()
min_word_count = train['word_count'].min()
mode_word_count = train['word_count'].mode()
median_word_count = train['word_count'].median()


print(f"Average Number of Words: {average_word_count:.2f}")
print(f"Maximum Number of Words: {max_word_count}")
print(f"Minimum Number of Words: {min_word_count}")
print(f"Mode of Number of Words: {mode_word_count}")
print(f"Median of Number of Words: {median_word_count}")

# Create a boxplot for fun
plt.figure(figsize=(10, 6))
sns.boxplot(x=train['word_count'])
plt.title('Boxplot')
plt.xlabel('Number of Words')
plt.show()

# %%
#Decide Vocab Size
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['text'])
vocab_size = len(tokenizer.word_index) + 1

print(vocab_size)

# %%
#Decide Max sequence length
max_sequence_length = max(len(sequence) for sequence in train['text'])
print(max_sequence_length)

# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train_sequences = tokenizer.texts_to_sequences(train['text'])
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length, padding='post')

y_train = y_train = train['generated']

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, GlobalMaxPooling1D

# Define the model
model = Sequential()

# Input layer with embedding for variable sequence length
embedding_dim = 30
model.add(Embedding(input_dim = vocab_size, output_dim = embedding_dim))

# Bidirectional LSTM layers
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(Bidirectional(LSTM(units=32, return_sequences=True)))

# Global Max Pooling layer 
model.add(GlobalMaxPooling1D())

# Dense layers for binary classification
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()



# %%


# Fit Model to training data
model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_split=0.2)



