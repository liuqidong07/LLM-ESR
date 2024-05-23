# here put the import lib
import numpy as np
import os
import torch
import torch.nn as nn
import pickle
from models.BaseModel import BaseSeqModel
from models.utils import PointWiseFeedForward



class SASRecBackbone(nn.Module):

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

        #timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,)
                                                    #   attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats



class SASRec(BaseSeqModel):
    
    def __init__(self, user_num, item_num, device, args):
        
        super(SASRec, self).__init__(user_num, item_num, device, args)

        # self.user_num = user_num
        # self.item_num = item_num
        # self.dev = device

        self.item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.backbone = SASRecBackbone(device, args)

        self.loss_func = torch.nn.BCEWithLogitsLoss()

        # self.filter_init_modules = []
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        item_seq_emb = self.item_emb(log_seqs)

        return item_seq_emb


    def log2feats(self, log_seqs, positions):
        '''Get the representation of given sequence'''
        seqs = self._get_embedding(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)

        log_feats = self.backbone(seqs, log_seqs)

        return log_feats


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions): # for training        
        '''Used to calculate pos and neg logits for loss'''
        log_feats = self.log2feats(seq, positions) # (bs, max_len, hidden_size)
        log_feats = log_feats[:, -1, :].unsqueeze(1) # (bs, hidden_size)

        pos_embs = self._get_embedding(pos.unsqueeze(1)) # (bs, 1, hidden_size)
        neg_embs = self._get_embedding(neg) # (bs, neg_num, hidden_size)

        pos_logits = torch.mul(log_feats, pos_embs).sum(dim=-1) # (bs, 1)
        neg_logits = torch.mul(log_feats, neg_embs).sum(dim=-1) # (bs, neg_num)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)
        indices = (pos != 0)    # do not calculate the padding units
        pos_loss, neg_loss = self.loss_func(pos_logits[indices], pos_labels[indices]), self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss

        return loss # loss


    def predict(self,
                seq, 
                item_indices, 
                positions,
                **kwargs): # for inference
        '''Used to predict the score of item_indices given log_seqs'''
        log_feats = self.log2feats(seq, positions) # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        item_embs = self._get_embedding(item_indices) # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits # preds # (U, I)
    

    def get_user_emb(self,
                     seq,
                     positions,
                     **kwargs):
        log_feats = self.log2feats(seq, positions) # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        
        return final_feat



class SASRec_seq(SASRec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        '''apply the seq-to-seq loss'''
        log_feats = self.log2feats(seq, positions)
        pos_embs = self._get_embedding(pos) # (bs, max_seq_len, hidden_size)
        neg_embs = self._get_embedding(neg) # (bs, max_seq_len, hidden_size)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)
        indices = (pos != 0)    # do not calculate the padding units
        pos_loss, neg_loss = self.loss_func(pos_logits[indices], pos_labels[indices]), self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss

        return loss




