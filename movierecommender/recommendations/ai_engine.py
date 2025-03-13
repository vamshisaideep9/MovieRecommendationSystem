import torch
from transformers import BertTokenizer, BertModel
import numpy as np


#Load BERT MODEL and Tokenizer 
hf_model_path = 'google-bert/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(hf_model_path)
model = BertModel.from_pretrained(hf_model_path)


def get_bert_embedding(text):
    """ Convert text into a BERT embedding vector """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get mean pooling of last hidden state
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.astype(np.float32) #convert to float32 for efficient storage