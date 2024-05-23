# here put the import lib
import torch.nn as nn



class BaseSeqModel(nn.Module):

    def __init__(self, user_num, item_num, device, args) -> None:
        
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = device
        self.freeze_modules = []
        self.filter_init_modules = []    # all modules should be initialized

    
    def _freeze(self):
        
        for name, param in self.named_parameters():
            try:
                flag = False    
                for fm in self.freeze_modules:
                    if fm in name:  # if the param in freeze_modules, freeze it
                        flag = True
                if flag:
                    param.requires_grad = False
            except:
                pass


    def _init_weights(self):
        
        for name, param in self.named_parameters():
            try:
                flag = True     # denote initialize this param 
                for fm in self.filter_init_modules:
                    if fm in name:  # if the param in filter_modules, do not initialize
                        flag = False    
                if flag:
                    nn.init.xavier_normal_(param.data)
            except:
                pass


    def _get_embedding(self, log_seqs):

        raise NotImplementedError("The function for sequence embedding is missed")



