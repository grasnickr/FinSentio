from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
config = AutoConfig.from_pretrained(model_name)

id2label = config.id2label
label2id = {v: k for k, v in id2label.items()}

def get_finbert_score(text:str) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Score: +1 * positive + (-1) * negative + 0 * neutral
    score = probs[label2id["positive"]] * 1 + probs[label2id["negative"]] * -1
    return round(score, 4)  
