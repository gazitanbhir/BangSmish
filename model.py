# model.py
import torch
import torch.nn as nn
from transformers import BertModel

class CNNCharacterLevel(nn.Module):
    def __init__(self, alphabet_size, embed_dim, num_filters, filter_sizes):
        super(CNNCharacterLevel, self).__init__()
        self.embedding = nn.Embedding(alphabet_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (f_size, embed_dim))
            for f_size in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (batch_size, 1, max_len, embed_dim)
        x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [nn.functional.max_pool1d(feature, feature.size(2)).squeeze(2) for feature in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return x

class CNN_BERT_Combined(nn.Module):
    def __init__(self, bert_model, cnn_char_model, hidden_dim, num_labels):
        super(CNN_BERT_Combined, self).__init__()
        self.bert = bert_model
        self.cnn_char = cnn_char_model
        self.fc = nn.Linear(self.bert.config.hidden_size + cnn_char_model.convs[0].out_channels * len(cnn_char_model.convs), hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask, char_input):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_output = bert_output.last_hidden_state[:, 0, :]  # Extract the [CLS] token
        char_output = self.cnn_char(char_input)
        combined = torch.cat((bert_cls_output, char_output), dim=1)
        combined = self.fc(combined)
        logits = self.classifier(combined)
        return logits

def load_model():
    from transformers import BertModel
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
    
    alphabet_size = 2032
    embed_dim = 32
    num_filters = 64
    filter_sizes = [3, 4, 5]
    
    cnn_char_model = CNNCharacterLevel(alphabet_size, embed_dim, num_filters, filter_sizes)
    model = CNN_BERT_Combined(bert_model, cnn_char_model, hidden_dim=128, num_labels=3)
    
    # Load the saved model weights with CPU mapping
    model.load_state_dict(torch.load('asstes/cnn_bert_model_state_dict.pth', map_location=torch.device('cpu')))
    model.eval()
    
    return model
