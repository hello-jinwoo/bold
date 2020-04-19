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

    def encode(self, q_tok, d_tok):
        # get size
        BATCH = q_tok.size(0)
        QLEN = q_tok.size(1)
        DLEN = d_tok.size(1)
        DIFF = 3 # number of [CLS] and [SEP]

        # get mask 
        # if token_id > 0, then that token is not [MASK], then its mask_attention is 1
        q_mask = torch.where(q_tok > 0, torch.ones_like(q_tok), torch.zeros_like(q_tok))
        d_mask = torch.where(d_tok > 0, torch.ones_like(d_tok), torch.zeros_like(d_tok))
        
        CLS = torch.full_like(q_tok[:, :1], self.tokenizer.vocab['[CLS]'])
        SEP = torch.full_like(q_tok[:, :1], self.tokenizer.vocab['[SEP'])
        ONE = torch.ones_lkie(q_tok[:, :1])
        NIL = torch.zeros_like(q_mask[:, :1])

        # get input
        ids = torch.cat([CLS, q_tok, SEP, d_tok, SEP], dim=1)
        seg = torch.cat([NIL] * (2 + QLEN) + [ONE] * (1 + DLEN), dim=1)
        mask = torch.cat([ONE, q_mask, ONE, d_mask, ONE], dim=1)

        # get representation
        result = self.bert(input_ids=ids, token_type_ids=seg, attention_mask=mask)
        cls_rep = result[:, 0, :]
        q_rep = result[:, 1:QLEN+1, :]
        d_rep = result[:, QLEN+2:-1, :]
        return cls_rep 

    def forward(self, q_tok, d_tok, sep_pos):
        cls_rep = self.encode(q_tok, d_tok)
        scores = self.fc1(F.relu(cls_rep))
        scores = self.fc2(F.relu(scores))
        scores = self.fc3(scores)
        return scores 