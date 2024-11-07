import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import os
from collections import Counter
import re
from langchain_community.document_loaders import WebBaseLoader
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    st.error(f"Error downloading NLTK data: {str(e)}")

# Constants
MAX_SEQUENCE_LENGTH = 5000
MAX_NUM_WORDS = 25000  # Match the pre-trained model's vocabulary size
EMBEDDING_DIM = 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_vocabulary(vocab_path):
    """
    Build a vocabulary that matches the size of the pre-trained model.
    """
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            return json.load(f)
    
    # Initialize with special tokens
    word2idx = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<START>': 2,
        '<END>': 3
    }
    
    # Generate placeholder vocabulary to match pre-trained model size
    for i in range(4, MAX_NUM_WORDS):
        word2idx[f'word_{i}'] = i
    
    # Save vocabulary
    try:
        with open(vocab_path, 'w') as f:
            json.dump(word2idx, f, indent=4)
    except Exception as e:
        st.error(f"Error saving vocabulary: {str(e)}")
    
    return word2idx

# Define the Fake News Detector model
class FakeNewsDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_length):
        super(FakeNewsDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(embedding_dim, 128, kernel_size=5)
        self.dense1 = nn.Linear(128, 128)
        self.dense2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1d(x))
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        return x

class TextPreprocessor:
    def __init__(self, vocab_path):
        self.word2idx = build_vocabulary(vocab_path)
        self.max_len = MAX_SEQUENCE_LENGTH
        
    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
        
    def transform(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
            
        sequences = []
        for text in texts:
            # Clean and tokenize text
            cleaned_text = self.clean_text(str(text))
            words = word_tokenize(cleaned_text)
            
            # Convert words to indices, using UNK token (1) for unknown words
            sequence = [1] * len(words)  # Initialize all words as UNK
            
            # Truncate or pad sequence
            if len(sequence) > self.max_len:
                sequence = sequence[:self.max_len]
            else:
                sequence = [0] * (self.max_len - len(sequence)) + sequence
            
            sequences.append(sequence)
        
        return torch.LongTensor(sequences)

def load_model_and_preprocessor(model_path, vocab_path):
    """Load the pre-trained model and initialize preprocessor."""
    try:
        # First build/load vocabulary
        word2idx = build_vocabulary(vocab_path)
        
        # Create the model with the correct vocabulary size
        model = FakeNewsDetector(
            vocab_size=MAX_NUM_WORDS,
            embedding_dim=EMBEDDING_DIM,
            max_seq_length=MAX_SEQUENCE_LENGTH
        ).to(DEVICE)
        
        # Load the pre-trained weights
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        # Create the preprocessor
        preprocessor = TextPreprocessor(vocab_path)
        
        return model, preprocessor
        
    except Exception as e:
        st.error(f"Error with model: {str(e)}")
        return None, None

def webscrape(link):
    try:
        loader = WebBaseLoader(link)
        data = loader.load()

        cleaned_data = ""
        for document in data:
            # Clean the page content
            cleaned_text = document.page_content.replace('\n', '').replace('\\n', ' ')
            
            # Filter for valid ASCII characters only
            cleaned_text = ''.join(char for char in cleaned_text if ord(char) < 128)
            
            # Append to the cleaned_data string
            cleaned_data += cleaned_text

        return cleaned_data
    except Exception as e:
        st.error(f"Error scraping website: {str(e)}")
        return ""

def run():
    st.title("Text or Website Content Verification")
    data=""
    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor('fake_news_detector.pth', 'vocab.json')
    
    if model is None or preprocessor is None:
        st.error("Failed to initialize the model. Please check the error messages above.")
        return

    # Text Verification Section
    st.header("Enter Text for Verification")
    
    # Large input box for text entry
    user_input = st.text_area("Enter text for verification", height=300)

    if st.button("Verify Text"):
        if user_input:
            try:
                # Clean and preprocess text
                input_sequence = preprocessor.transform([user_input]).to(DEVICE)

                # Get prediction
                with torch.no_grad():
                    output = model(input_sequence)
                    prediction = torch.sigmoid(output).item()
                
                # Determine result and confidence
                result = "Real News" if prediction > 0.5 else "Fake News"
                confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
                #data+=f"The current article here is {result}"
                #st.write(output)
                
                # Display results
                st.subheader("Results:")
                st.write(f"Classification: **{result}**")
                st.write(f"Confidence: {confidence:.2f}%")
                #data+=f'Classification: **{result}* Confidence: {confidence:.2f}%'
                # Display preprocessed text
                st.subheader("Preprocessed Text:")
                st.text_area("", preprocessor.clean_text(user_input), height=150, disabled=True)
                
            except Exception as e:
                st.error(f"Error processing text: {str(e)}")
        else:
            st.warning("Please enter some text.")

    # Website Link Verification Section
    st.header("Enter Website URL for Scraping")
    
    # Input for website link
    website_url = st.text_input("Website URL", "")

    if st.button("Scrape Website"):
        if website_url:
            scraped_content = webscrape(website_url)
            if scraped_content:
                try:
                    input_sequence = preprocessor.transform([scraped_content]).to(DEVICE)

                    # Get prediction
                    with torch.no_grad():
                        output = model(input_sequence)
                        prediction = torch.sigmoid(output).item()
                    
                    # Determine result and confidence
                    result = "Real News" if prediction > 0.5 else "Fake News"
                    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
                    
                    # Display results
                    st.subheader("Results:")
                    st.write(f"Classification: **{result}**")
                    st.write(f"Confidence: {confidence:.2f}%")
                   # data+=f'Classification: **{result}* Confidence: {confidence:.2f}%'
                    # Display scraped content
                    st.subheader("Scraped Content:")
                    st.text_area("", scraped_content[:1000] + "...", height=150, disabled=True)
                    
                except Exception as e:
                    st.error(f"Error processing scraped content: {str(e)}")
        else:
            st.warning("Please enter a valid website URL.")
    #return f"These are the analysis of current arcticle or text analysis {data}\nOnly answer when asked about it ."

if __name__ == "__main__":
    run()