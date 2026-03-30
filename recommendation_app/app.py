import pickle
import numpy as np
import gradio as gr
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

model = load_model("gru_model.keras", compile=False)

with open("user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)
with open("item_encoder.pkl", "rb") as f:
    item_encoder = pickle.load(f)
with open("user_sequences.pkl", "rb") as f:
    user_sequences = pickle.load(f)

all_items = [item for seq in user_sequences.values() for item in seq]
item_counts = Counter(all_items)
threshold = int(0.2 * len(item_counts))
popular_items = set([item for item, _ in item_counts.most_common(threshold)])
max_len = 30

def recommend(user_input):
    try:
        user_id = int(user_input)
        if user_id not in user_sequences:
            return f"Invalid User ID. Enter 0 to {len(user_sequences)-1}"
        seq = user_sequences[user_id][-max_len:]
        padded = pad_sequences([seq], maxlen=max_len, padding='pre')
        preds = model.predict(padded, verbose=0)[0]
        top_k = preds.argsort()[-10:][::-1]
        popular_count = 0
        output = ""
        for i, item in enumerate(top_k, 1):
            product = item_encoder.inverse_transform([item])[0]
            tag = "Popular" if item in popular_items else "Long-tail"
            if item in popular_items:
                popular_count += 1
            output += f"{i}. {product} [{tag}]\n"
        output += f"\nPopularity Bias: {popular_count * 10}%\n"
        output += f"Long-tail Exposure: {(10 - popular_count) * 10}%"
        return output
    except Exception as e:
        return f"Error: {str(e)}"

app = gr.Interface(fn=recommend, inputs="text", outputs="text", title="Recommendation System")
app.launch()