from flair.data import Sentence
from flair.nn import Classifier
import flair
import torch


flair.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tagger = Classifier.load('sentiment')

def get_flair_score(text: str) -> float:
    
    sentence = Sentence(text)
    
    tagger.predict(sentence)
    

    label = sentence.labels[0].value
    confidence = sentence.labels[0].score
    

    if label == 'POSITIVE':
        score = confidence
    elif label == 'NEGATIVE':
        score = -confidence
    else:
        score = 0.0
        
    return round(score, 4)

#Test the function
if __name__ == "__main__":
    test_text = "The company reported record profits for Q3, massively beating analyst expectations."
    print(f"Test Nachricht: {test_text}")
    print(f"Flair Sentiment Score: {get_flair_score(test_text)}")