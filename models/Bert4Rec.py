# here put the import lib
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import *
from models.BaseModel import BaseSeqModel



class BertBackbone(nn.Module):

    def __init__(self, device, args) -> None:
        
        super().__init__()

        self.dev = device
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)

        for _ in range(args.trm_num):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_size,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_size, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    
    def forward(self, seqs, log_seqs):

        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats
    


class Bert4Rec(BaseSeqModel):

    def __init__(self, user_num, item_num, device, args):
        
        super(Bert4Rec, self).__init__(user_num, item_num, device, args)

        self.mask_token = item_num + 1
        self.item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.backbone = BertBackbone(self.dev, args)

        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self._init_weights()


    def _get_embedding(self, log_seqs):

        item_seq_emb = self.item_emb(log_seqs)

        return item_seq_emb


    def log2feats(self, log_seqs, positions):

        seqs = self._get_embedding(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)

        log_feats = self.backbone(seqs, log_seqs)

        return log_feats


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs): # for training        

        log_feats = self.log2feats(seq, positions) # (bs, max_len, hidden_size)
        mask_index = torch.where(pos>0)
        log_feats = log_feats[mask_index] # (bs, mask_num, hidden_size)

        pos_embs = self._get_embedding(pos) # (bs, mask_num, hidden_size)
        neg_embs = self._get_embedding(neg) # (bs, mask_num, hidden_size)
        pos_embs = pos_embs[mask_index]
        neg_embs = neg_embs[mask_index]

        pos_logits = torch.mul(log_feats, pos_embs).sum(dim=-1) # (bs, mask_num)
        neg_logits = torch.mul(log_feats, neg_embs).sum(dim=-1) # (bs, mask_num)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)
        pos_loss, neg_loss = self.loss_func(pos_logits, pos_labels), self.loss_func(neg_logits, neg_labels)
        loss = pos_loss + neg_loss

        return loss # loss


    def predict(self, 
                seq, 
                item_indices, 
                positions,
                **kwargs): # for inference

        log_seqs = torch.cat([seq, self.mask_token * torch.ones(seq.shape[0], 1, device=self.dev)], dim=1)
        pred_position = positions[:, -1] + 1
        positions = torch.cat([positions, pred_position.unsqueeze(1)], dim=1)
        log_feats = self.log2feats(log_seqs[:, 1:].long(), positions[:, 1:].long()) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        item_embs = self._get_embedding(item_indices) # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits # preds # (U, I)



