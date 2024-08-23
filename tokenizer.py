# tokenizer.py
from transformers import BertTokenizer
import joblib

def load_tokenizer_and_vocab():
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    label_encoder = joblib.load('asstes/label_encoder.pkl')
    char_vocab = joblib.load('asstes/char_vocab.pkl')[0]
    return tokenizer, label_encoder, char_vocab

def tokenize_function(text, tokenizer):
    return tokenizer(text, padding='max_length', truncation=True, max_length=128)

def char_tokenizer(text, char_vocab, max_len=256):
    encoded = [char_vocab.get(c, char_vocab.get('<UNK>', 0)) for c in text[:max_len]]
    encoded += [char_vocab.get('<PAD>', 0)] * (max_len - len(encoded))
    return encoded
