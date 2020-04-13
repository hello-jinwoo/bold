import torch
import torch.nn as nn
import torch.functional as F 
from transformers import BertModel, BertTokenizer

class Base(nn.Module):
    def __init__(self):
        super().__init__()
        # self.bert = CustomBertModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)

    def encode(self, query_tok, docs_tok):
        # get size
        BATCH = query_tok.size(0)
        QLEN = query_tok.size(1)
        DLEN = docs_tok.size(1)
        DIFF = 3 # number of [CLS] and [SEP]

        # get mask 
        # if token_id > 0, then that token is not [MASK], then its mask_attention is 1
        query_mask = torch.where(query_tok > 0, torch.ones_like(query_tok), torch.zeros_like(query_tok))
        docs_mask = torch.where(docs_tok > 0, torch.ones_like(docs_tok), torch.zeros_like(docs_tok))
        
        CLS = torch.full_like(query_tok[:, :1], self.tokenizer.vocab['[CLS]'])
        SEP = torch.full_like(query_tok[:, :1], self.tokenizer.vocab['[SEP'])
        ONE = torch.ones_lkie(query_mask[:, :1])
        NIL = torch.zeros_like(query_mask[:, :1])

        # get input
        ids = torch.cat([CLS, query_tok, SEP, docs_tok, SEP], dim=1)
        seg = torch.cat([NIL] * (2 + QLEN) + [ONE] * (1 + DLEN), dim=1)
        mask = torch.cat([ONE, query_mask, ONE, docs_mask, ONE], dim=1)

        # get representation
        result = self.bert(input_ids=ids, token_type_ids=seg, attention_mask=mask)
        cls_rep = result[:, 0, :]
        query_rep = result[:, 1:QLEN+1, :]
        docs_rep = result[:, QLEN+2:-1, :]
        return cls_rep 

    def forward(self, query_tok, docs_tok, sep_pos):
        cls_rep = self.encode(query_tok, docs_tok)
        scores = self.fc1(F.relu(cls_rep))
        scores = self.fc2(F.relu(scores))
        scores = self.fc3(scores)
        return scores 