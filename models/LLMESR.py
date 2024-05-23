# here put the import lib
import torch
import torch.nn as nn
from models.DualLLMSRS import DualLLMSASRec, DualLLMGRU4Rec, DualLLMBert4Rec
from models.utils import Contrastive_Loss2



class LLMESR_SASRec(DualLLMSASRec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        
        log_feats = self.log2feats(seq, positions)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq, sim_positions)[:, -1, :]    # (bs*sim_num, hidden_size)
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        if self.user_sim_func == "cl":
            # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        loss += self.alpha * align_loss

        return loss
    


class LLMESR_GRU4Rec(DualLLMGRU4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        
        log_feats = self.log2feats(seq)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq)[:, -1, :]    # (bs*sim_num, hidden_size)
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        if self.user_sim_func == "cl":
            # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        loss += self.alpha * align_loss

        return loss



class LLMESR_Bert4Rec(DualLLMBert4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        
        log_feats = self.log2feats(seq, positions)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq, sim_positions)[:, -1, :]
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        if self.user_sim_func == "cl":
            # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        loss += self.alpha * align_loss

        return loss



