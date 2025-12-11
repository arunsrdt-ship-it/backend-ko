from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- LOAD BRAIN (Runs once when server starts) ---
print("Loading model and memory...")
# Load the AI model (Cached to avoid re-downloading)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load Data
base_dir = os.path.dirname(os.path.abspath(__file__))
memory_path = os.path.join(base_dir, 'bot_memory.pkl')
responses_path = os.path.join(base_dir, 'intent_responses.json')

with open(memory_path, 'rb') as f:
    data = pickle.load(f)

with open(responses_path, 'r') as f:
    responses = json.load(f)

def get_bot_response(user_query):
    # 1. Understand
    query_embedding = embedder.encode([user_query])
    
    # 2. Match
    similarities = cosine_similarity(query_embedding, data['embeddings'])
    best_idx = np.argmax(similarities)
    score = similarities[0][best_idx]
    intent = data['intents'][best_idx]

    # 3. Threshold (0.55)
    if score < 0.55:
        return "I'm not sure I understand. Could you try asking in a different way?"
    
    # 4. Reply
    return responses.get(intent, "I don't have info on that yet.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def get_bot_response_route():
    userText = request.args.get('msg')
    response = get_bot_response(userText)
    return str(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)