# here put the import lib
import torch
import torch.nn as nn
from models.BaseModel import BaseSeqModel



class GRU4RecBackbone(nn.Module):

    def __init__(self, device, args) -> None:
        
        super().__init__()

        self.dev = device
        self.gru = nn.GRU(
            input_size=args.hidden_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            bias=False,
            batch_first=True
        )


    def forward(self, seqs, log_seqs):

        log_feats, _ = self.gru(seqs)

        return log_feats



class GRU4Rec(BaseSeqModel):

    def __init__(self, user_num, item_num, device, args) -> None:
        
        super(GRU4Rec, self).__init__(user_num, item_num, device, args)

        self.dev = device
        self.item_emb = nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        self.backbone = GRU4RecBackbone(device, args)

        self.loss_func = nn.BCEWithLogitsLoss()

        self._init_weights()


    def _get_embedding(self, log_seqs):

        item_seq_emb = self.item_emb(log_seqs)

        return item_seq_emb

    
    def log2feats(self, log_seqs):

        seqs = self.item_emb(log_seqs)
        mask = (log_seqs > 0).unsqueeze(-1)
        # seqs *= mask    # the padding input is 0
        log_feats = self.backbone(seqs, log_seqs)

        return log_feats


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        # inputs: (bs, max_seq_len, hidden_size), mask: (bs, max_seq_len)
        log_feats = self.log2feats(seq)
        log_feats = log_feats[:, -1, :].unsqueeze(1)

        pos_embs = self._get_embedding(pos.unsqueeze(1)) # (bs, 1, hidden_size)
        neg_embs = self._get_embedding(neg) # (bs, neg_num, hidden_size)

        pos_logits = torch.mul(log_feats, pos_embs).sum(dim=-1) # (bs, 1)
        neg_logits = torch.mul(log_feats, neg_embs).sum(dim=-1) # (bs, neg_num)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)

        indices = (pos != 0)    # do not calculate the padding units
        pos_loss, neg_loss = self.loss_func(pos_logits[indices], pos_labels[indices]), self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss

        return loss
    

    def predict(self, 
                seq, 
                item_indices, 
                positions,
                **kwargs): # for inference
        '''Used to predict the score of item_indices given log_seqs'''
        log_feats = self.log2feats(seq)
        final_feat = log_feats[:, -1, :]
        item_embs = self._get_embedding(item_indices) # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits # preds # (U, I)



class GRU4Rec_seq(GRU4Rec):

    def __init__(self, user_num, item_num, device, args) -> None:
        
        super().__init__(user_num, item_num, device, args)


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        '''apply the seq-to-seq loss'''
        log_feats = self.log2feats(seq)
        pos_embs = self._get_embedding(pos) # (bs, max_seq_len, hidden_size)
        neg_embs = self._get_embedding(neg) # (bs, max_seq_len, hidden_size)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)
        indices = (pos != 0)    # do not calculate the padding units
        pos_loss, neg_loss = self.loss_func(pos_logits[indices], pos_labels[indices]), self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss

        return loss

