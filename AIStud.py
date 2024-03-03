# %% [markdown]
# **Oakes Benchmark 6: Data Product Completion**

# %% [markdown]
# To start out, it's important to have the following installed: 
# 
# We'll use pip install, however for my environment they will show 'already satisfied' in an extremely long brick of text, so for this exercise I will put them all in one kernel and not display the output. Trust that it worked, as I use these libraries in the code.

# %%
#Install and Import Necessary Libraries

#pip install NLTK 
#pip install pandas 
#pip install requests 
#pip install tensorflow
#pip install wordcloud
#pip install seaborn
#pip install streamlit

# %%
#Import packages / libraries to follow naming conventions

#One-off
import pandas as pd
import requests
import numpy as np
import streamlit as st

#NLTK Specific
import nltk
import tensorflow as tf
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
import json
import AIStudeModel


# %% [markdown]
# Everything is installed and set up; now I'll start preparing the needed functions for the scripts necessary activites.

# %%
#Define Functions for Getting File

# dataframe from a csv
def GetFile(csv_path):
    df = pd.read_csv(csv_path, encoding='UTF-8')
    return df

#dataframe from a url
def GetFileURL(url):
    response = requests.get(url)
    df = pd.read_csv(pd.compat.StringIO(response.text), encoding='UTF-8')
    return df

# %%
#Define function for inital data exploration

def Explore (dataframe):
    
    #Printing basic data information
    print("Basic Information about the DataFrame:")
    print(dataframe.info())
    print("\nFirst 5 Rows of the DataFrame:")
    print(dataframe.head())

# Define Function for relevant data visualization
    
#def LookPretty(dataframe):
    
    #Prep text for wordcloud
   # text_corpus = ' '.join(dataframe[text_column].astype(str))
    #wordcloud = WordCloud(width=800, height=400, random_state=42, max_font_size=100).generate(text_corpus)

    #Wordcloud of Text
   # plt.figure(figsize=(12, 8))
   # plt.imshow(wordcloud, interpolation="bilinear")
   # plt.axis('off')
  #  plt.title('Word Cloud for Essay')
  #  plt.show()

     # Display plot of stopwords
    # plt.figure(figsize=(10, 6))
    # sns.histplot(dataframe['stopwords_count'], kde=True)
    # plt.title('Distribution of Stopwords Count')
    # plt.xlabel('Stopwords Count')
    # plt.ylabel('Frequency')
    # plt.show()

    #Display a plot of stopwords
    #plot_values = np.histogram(dataframe['stop_words_count'])
    #st.bar_chart(plot_values)

# %%
#Define Function for Cleaning   

#stop word setup
def CleanSweep(dataframe):
    stop_words = set(stopwords.words('english'))
    # Convert to lowercase and remove punctuation
    cleaned_text = ''.join([char for char in dataframe['text'] if char.isalnum() or char.isspace()])
    
    # Count the number of stop words
    #stop_words_count = sum(1 for word in words if word in stop_words)
    
    # Remove stop words
    #filtered_words = [word for word in words if word not in stop_words]

    # Tokenize and pad sequences using the same tokenizer and max_length
    with open('tokenizer.json', 'r', encoding='UTF-8') as json_file:
        loaded_tokenizer_json = json_file.read()
        loaded_tokenizer = tokenizer_from_json(loaded_tokenizer_json)

    sequences = loaded_tokenizer.texts_to_sequences(cleaned_text)
    padded_sequences = pad_sequences(sequences, maxlen=8452, padding='post')
    
    padded_sequences = pd.DataFrame(padded_sequences)

    return padded_sequences




# %%
#Initiate app through Streamlit
#streamlit run app.py

# %%
#Defining the app and functionalities

# Main function to run the app
def main():
    st.title("Educational Data Product")

    # File upload section
    st.sidebar.header("Upload CSV File")
    st.sidebar.subheader("CSV must have UTF-8 encoding so that streamlit can read it.")
    st.sidebar.subheader("Open in Notebook > Save As > Encoding drop down > UTF-8")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])


    if uploaded_file is not None:
        # Process the uploaded file
        st.header("Uploaded Data Preview:")
        df = GetFile(uploaded_file)
        st.write(df)

        # Clean things up
        df = CleanSweep(df)

        # Check Out the File
        Explore(df)

        # Import model and run
        st.header("Model Results:")

        # Make predictions with user data
        predictions = AIStudeModel.model.predict(df)

        # Display results
        df['predictions'] = predictions
        st.write("Predictions:")
        st.write(df[['text', 'predictions']])


if __name__ == "__main__":
    main()


# %% [markdown]
# Help Functions
# 
# In the next Benchmark, Implementation is the goal. During implementation, there is a streamlit functionality to add a navigation bar on the side. I plan to have two pages: Main and Help.
# 
# The Help Page will have the following:
# 
# - instrcutions for csv files
# - overview of code
# - used libraries
# - example file
# - screenshots or a video of a user walking through the process