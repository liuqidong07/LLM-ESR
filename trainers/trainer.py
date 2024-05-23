# here put the import lib
import os
import torch
from tqdm import trange
from utils.earlystop import EarlyStoppingNew
from utils.utils import get_n_params
from models.LLMESR import LLMESR_SASRec, LLMESR_Bert4Rec, LLMESR_GRU4Rec



class Trainer(object):

    def __init__(self, args, logger, writer, device, generator):

        self.args = args
        self.logger = logger
        self.writer = writer
        self.device = device
        self.user_num, self.item_num = generator.get_user_item_num()
        self.start_epoch = 0    # define the start epoch for keepon training

        self.logger.info('Loading Model: ' + args.model_name)
        self._create_model()
        logger.info('# of model parameters: ' + str(get_n_params(self.model)))

        self._set_optimizer()
        self._set_scheduler()
        self._set_stopper()

        if args.keepon:
            self._load_pretrained_model()

        self.loss_func = torch.nn.BCEWithLogitsLoss()
        
        self.train_loader = generator.make_trainloader()
        self.valid_loader = generator.make_evalloader()
        self.test_loader = generator.make_evalloader(test=True)
        self.generator = generator

        # get item pop and user len
        self.item_pop = generator.get_item_pop()
        self.user_len = generator.get_user_len()

        #self.watch_metric = 'NDCG@10'  # use which metric to select model
        self.watch_metric = args.watch_metric

    
    def _create_model(self):
        '''create your model'''
        if self.args.model_name == "llmesr_sasrec":
            self.model = LLMESR_SASRec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "llmesr_gru4rec":
            self.model = LLMESR_GRU4Rec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "llmesr_bert4rec":
            self.model = LLMESR_Bert4Rec(self.user_num, self.item_num, self.device, self.args)
        else:
            raise ValueError
        
        self.model.to(self.device)
    

    def _load_pretrained_model(self):

        self.logger.info("Loading the trained model for keep on training ... ")
        checkpoint_path = os.path.join(self.args.keepon_path, 'pytorch_model.bin')

        model_dict = self.model.state_dict()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        pretrained_dict = checkpoint['state_dict']

        # filter out required parameters
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # Print: how many parameters are loaded from the checkpoint
        self.logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        self.model.load_state_dict(model_dict)  # load model parameters
        self.optimizer.load_state_dict(checkpoint['optimizer']) # load optimizer
        self.scheduler.load_state_dict(checkpoint['scheduler']) # load scheduler
        self.start_epoch = checkpoint['epoch']  # load epoch

    
    def _set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.args.lr,
                                          weight_decay=self.args.l2,
                                          )

    
    def _set_scheduler(self):

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.args.lr_dc_step,
                                                         gamma=self.args.lr_dc)


    def _set_stopper(self):

        self.stopper = EarlyStoppingNew(patience=self.args.patience, 
                                     verbose=False,
                                     path=self.args.output_dir,
                                     trace_func=self.logger)


    def _train_one_epoch(self, epoch):

        return NotImplementedError
    

    def _prepare_train_inputs(self, data):
        """Prepare the inputs as a dict for training"""
        assert len(self.generator.train_dataset.var_name) == len(data)
        inputs = {}
        for i, var_name in enumerate(self.generator.train_dataset.var_name):
            inputs[var_name] = data[i]

        return inputs
    

    def _prepare_eval_inputs(self, data):
        """Prepare the inputs as a dict for evaluation"""
        inputs = {}
        assert len(self.generator.eval_dataset.var_name) == len(data)
        for i, var_name in enumerate(self.generator.eval_dataset.var_name):
            inputs[var_name] = data[i]

        return inputs


    def eval(self, epoch=0, test=False):

        return NotImplementedError


    def train(self):

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running training **********")
        self.logger.info("  Batch size = %d", self.args.train_batch_size)
        res_list = []
        train_time = []

        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.num_train_epochs), desc="Epoch"):

            t = self._train_one_epoch(epoch)
            
            train_time.append(t)

            # evluate on validation per 20 epochs
            if (epoch % 1) == 0:
                
                metric_dict = self.eval(epoch=epoch)
                res_list.append(metric_dict)
                #self.scheduler.step()
                self.stopper(metric_dict[self.watch_metric], epoch, model_to_save, self.optimizer, self.scheduler)

                if self.stopper.early_stop:

                    break
        
        best_epoch = self.stopper.best_epoch
        best_res = res_list[best_epoch - self.start_epoch]
        self.logger.info('')
        self.logger.info('The best epoch is %d' % best_epoch)
        self.logger.info('The best results are NDCG@10: %.5f, HR@10: %.5f' %
                    (best_res['NDCG@10'], best_res['HR@10']))
        
        res = self.eval(test=True)

        return res, best_epoch
    


    def test(self):
        """Do test directly. Set the output dir as the path that save the checkpoint"""
        res = self.eval(test=True)

        return res, -1



    def get_model(self):

        return self.model

    
    def get_model_param_num(self):

        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        freeze_num = total_num - trainable_num

        return freeze_num, trainable_num


