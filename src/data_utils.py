import pandas as pd
from transformers import AutoTokenizer
from sklearn.base import TransformerMixin, BaseEstimator

class PreProcessor(TransformerMixin, BaseEstimator):
    def __init__(self, model_name='bert-base-uncased', max_length=128)->None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_filtered = X.filter(lambda x: x['label'] != -1)
        premises = X_filtered['premise'].tolist()
        hypotheses = X_filtered['hypothesis'].tolist()
        encodings = self.tokenizer(premises, hypotheses, padding=True, truncation=True, max_length=self.max_length)
        return encodings
