# chatbot.py (DistilBERT)
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import json
import pickle
import joblib
import nltk
import torch
import requests
import contractions
import numpy as np
from openai import OpenAI
from pathlib import Path
from langid import classify
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

API_KEY = "" # Set your OpenAI API key here
client = OpenAI(api_key=API_KEY)

# === Load shortform dictionary ===
shortform_dict_path = os.path.join(os.path.dirname(__file__), "shortform_dict.json")
with open(shortform_dict_path, "r") as f:
    shortform_dict = json.load(f)

# === Text Preprocessing ===
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def expand_shortforms(text, shortform_dict):
    def replace(match):
        word = match.group(0)
        return shortform_dict.get(word.lower(), word)
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in shortform_dict.keys()) + r')\b', flags=re.IGNORECASE)
    return pattern.sub(replace, text)

def clean_text(text):
    text = re.sub(r'\[name\]', '', text)
    text = re.sub(r'[-—]', ' ', text)
    text = re.sub(r'[^a-zA-Z\s\.\!\?]', '', text)
    return ' '.join(text.split())

def handle_negations(tokens):
    negation_words = {"not", "no", "never", "n't", "neither", "nor"}
    new_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] in negation_words and i + 1 < len(tokens):
            new_tokens.append(tokens[i] + "_" + tokens[i + 1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess(text):
    """
    Text preprocessing pipeline:
    1. Convert to lowercase
    2. Fix contractions
    3. Expand shortforms
    4. Clean text (remove special chars, etc.)
    5. Tokenize
    6. Handle negations
    7. Lemmatize
    8. Remove stop words
    """
    text = text.lower()
    text = contractions.fix(text)
    text = expand_shortforms(text, shortform_dict)
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = handle_negations(tokens)
    lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    filtered_tokens = [word for word in lemmatized if word not in stop_words]
    return " ".join(filtered_tokens)

# === Mixed-Language Translation ===
def translate_mixed_language(text, target_language="English", api_key=API_KEY):
    # Detect language
    try:
        lang, _ = classify(text)
    except Exception as e:
        lang = "unknown"
        print(f"⚠️ Language detection failed: {e}")

    # If already English, skip translation
    if lang == "en":
        return text

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",  # or use "gpt-4o"
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional multilingual translator. "
                        "Translate the following mixed-language text into English, keeping the meaning accurate. "
                        "Preserve slang, proper nouns, and domain-specific terms. Return only the translated result."
                    )
                },
                {
                    "role": "user",
                    "content": f"Translate to {target_language}:\n\n{text}"
                }
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        translated_text = response.choices[0].message.content.strip()
        return translated_text

    except Exception as e:
        raise Exception(f"Translation failed: {str(e)}")

# === Load DistilBERT model ===
model_dir = Path(__file__).parent / "distilbert"
distilbert_model = DistilBertForSequenceClassification.from_pretrained(str(model_dir / "distilbert_model"))
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir / "distilbert_tokenizer"))
mlb = joblib.load(str(model_dir / "label_binarizer.pkl"))
emotion_columns = mlb.classes_

with open(model_dir / "metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
max_len = metadata["max_len"]

# === Emotion Prediction ===
def predict_emotions(text, threshold=0.15, api_key=API_KEY):
    """
    Predict emotions from preprocessed text using DistilBERT.
    Assumes text is already translated and preprocessed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distilbert_model.to(device)
    distilbert_model.eval()

    inputs = distilbert_tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    emotion_confidences = [
        (emotion_columns[i], float(prob))
        for i, prob in enumerate(probs)
        if prob >= threshold
    ]

    emotion_confidences.sort(key=lambda x: x[1], reverse=True)
    return emotion_confidences

# === Response Generation ===
def generate_response(user_input, emotion_confidences, conversation_history, api_key=API_KEY):
    """
    Generates empathetic response using OpenAI GPT-4.1-mini based on detected emotions
    """
    if not emotion_confidences:
        return "I'm here to listen. How are you feeling today?"

    primary_emotion, primary_confidence = emotion_confidences[0]

    # Compose the system prompt
    system_prompt = (
        "You are an empathetic counselor. "
        f"The user is primarily feeling {primary_emotion.capitalize()} ({primary_confidence:.0%} confidence). "
        "Respond with these guidelines:\n"
        "1. Start by acknowledging the emotion\n"
        "2. Show genuine understanding\n"
        "3. Keep response short (1-3 sentences)\n"
        "4. Use a warm, human-like tone\n"
        "5. If the user is feeling {primary_emotion.capitalize()}, respond with a message that is relevant to the emotion"
    )

    # Compose message history
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ] + conversation_history[-4:]  # Last 2 exchanges

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            temperature=max(0.4, min(primary_confidence, 0.8)),
            top_p=0.9,
            max_tokens=300,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"🚨 Response generation failed: {str(e)}")
        return "I'm here to listen. Could you tell me more about how you're feeling?"

# === Chatbot Interface ===
def start_chat():
    print("🤖 Chatbot: Hi! How are you feeling today? (type 'exit' to quit)")
    conversation_history = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("🤖 Chatbot: Goodbye! Take care of yourself.")
            break
        
        # Add to conversation history
        conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Emotion detection
            emotion_confidences = predict_emotions(user_input)
            
            # Generate emotion-aware response
            ai_response = generate_response(
                user_input=user_input,
                emotion_confidences=emotion_confidences,
                conversation_history=conversation_history
            )
            
            # Add AI response to history
            conversation_history.append({"role": "assistant", "content": ai_response})
            
            # Display with emotion confidence
            if emotion_confidences:
                conf_display = ", ".join([f"{e.capitalize()}({c:.0%})" for e, c in emotion_confidences])
                print(f"🤖 Chatbot [{conf_display}]: {ai_response}")
            else:
                print(f"🤖 Chatbot: {ai_response}")
                
        except Exception as e:
            print(f"🤖 Chatbot: Sorry, something went wrong - {str(e)}")
            # Fallback to simple response
            conversation_history.append({
                "role": "assistant", 
                "content": "I'm having trouble understanding. Could you rephrase?"
            })

if __name__ == "__main__":
    start_chat()