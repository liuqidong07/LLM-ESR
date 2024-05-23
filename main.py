# here put the import lib
import os
import argparse
import torch

from generators.generator import Seq2SeqGeneratorAllUser
from generators.generator import GeneratorAllUser
from generators.bert_generator import BertGeneratorAllUser
from trainers.sequence_trainer import SeqTrainer
from utils.utils import set_seed
from utils.logger import Logger


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--model_name", 
                    default='llmesr_sasrec',
                    choices=[
                    "llmesr_sasrec", "llmesr_bert4rec", "llmesr_gru4rec",
                    ],
                    type=str, 
                    required=False,
                    help="model name")
parser.add_argument("--dataset", 
                    default="yelp", 
                    choices=["yelp", "fashion", "beauty",],  # preprocess by myself
                    help="Choose the dataset")
parser.add_argument("--inter_file",
                    default="inter",
                    type=str,
                    help="the name of interaction file")
parser.add_argument("--demo", 
                    default=False, 
                    action='store_true', 
                    help='whether run demo')
parser.add_argument("--pretrain_dir",
                    type=str,
                    default="sasrec_seq",
                    help="the path that pretrained model saved in")
parser.add_argument("--output_dir",
                    default='./saved/',
                    type=str,
                    required=False,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--check_path",
                    default='',
                    type=str,
                    help="the save path of checkpoints for different running")
parser.add_argument("--do_test",
                    default=False,
                    action="store_true",
                    help="whehther run the test on the well-trained model")
parser.add_argument("--do_emb",
                    default=False,
                    action="store_true",
                    help="save the user embedding derived from the SRS model")
parser.add_argument("--do_group",
                    default=False,
                    action="store_true",
                    help="conduct the group test")
parser.add_argument("--keepon",
                    default=False,
                    action="store_true",
                    help="whether keep on training based on a trained model")
parser.add_argument("--keepon_path",
                    type=str,
                    default="normal",
                    help="the path of trained model for keep on training")
parser.add_argument("--clip_path",
                    type=str,
                    default="",
                    help="the path to save the CLIP-pretrained embedding and adapter")
parser.add_argument("--ts_user",
                    type=int,
                    default=10,
                    help="the threshold to split the short and long seq")
parser.add_argument("--ts_item",
                    type=int,
                    default=20,
                    help="the threshold to split the long-tail and popular items")

# Model parameters
parser.add_argument("--hidden_size",
                    default=64,
                    type=int,
                    help="the hidden size of embedding")
parser.add_argument("--trm_num",
                    default=2,
                    type=int,
                    help="the number of transformer layer")
parser.add_argument("--num_heads",
                    default=1,
                    type=int,
                    help="the number of heads in Trm layer")
parser.add_argument("--num_layers",
                    default=1,
                    type=int,
                    help="the number of GRU layers")
parser.add_argument("--cl_scale",
                    type=float,
                    default=0.1,
                    help="the scale for contastive loss")
parser.add_argument("--mask_crop_ratio",
                    type=float,
                    default=0.3,
                    help="the mask/crop ratio for CL4SRec")
parser.add_argument("--tau",
                    default=1,
                    type=float,
                    help="the temperature for contrastive loss")
parser.add_argument("--sse_ratio",
                    default=0.4,
                    type=float,
                    help="the sse ratio for SSE-PT model")
parser.add_argument("--dropout_rate",
                    default=0.5,
                    type=float,
                    help="the dropout rate")
parser.add_argument("--max_len",
                    default=200,
                    type=int,
                    help="the max length of input sequence")
parser.add_argument("--mask_prob",
                    type=float,
                    default=0.4,
                    help="the mask probability for training Bert model")
parser.add_argument("--aug",
                    default=False,
                    action="store_true",
                    help="whether augment the sequence data")
parser.add_argument("--aug_seq",
                    default=False,
                    action="store_true",
                    help="whether use the augmented data")
parser.add_argument("--aug_seq_len",
                    default=0,
                    type=int,
                    help="the augmented length for each sequence")
parser.add_argument("--aug_file",
                    default="inter",
                    type=str,
                    help="the augmentation file name")
parser.add_argument("--train_neg",
                    default=1,
                    type=int,
                    help="the number of negative samples for training")
parser.add_argument("--test_neg",
                    default=100,
                    type=int,
                    help="the number of negative samples for test")
parser.add_argument("--suffix_num",
                    default=5,
                    type=int,
                    help="the suffix number for augmented sequence")
parser.add_argument("--prompt_num",
                    default=2,
                    type=int,
                    help="the number of prompts")
parser.add_argument("--freeze",
                    default=False,
                    action="store_true",
                    help="whether freeze the pretrained architecture when finetuning")
parser.add_argument("--pg",
                    default="length",
                    choices=['length', 'attention'],
                    type=str,
                    help="choose the prompt generator")
parser.add_argument("--use_cross_att",
                    default=False,
                    action="store_true",
                    help="whether add a cross-attention to interact the dual-view")
parser.add_argument("--alpha",
                    default=0.1,
                    type=float,
                    help="the weight of auxiliary loss")
parser.add_argument("--user_sim_func",
                    default="kd",
                    type=str,
                    help="the type of user similarity function to derive the loss")
parser.add_argument("--item_reg",
                    default=False,
                    action="store_true",
                    help="whether regularize the item embedding by CL")
parser.add_argument("--beta",
                    default=0.1,
                    type=float,
                    help="the weight of regulation loss")
parser.add_argument("--sim_user_num",
                    default=10,
                    type=int,
                    help="the number of similar users for enhancement")
parser.add_argument("--split_backbone",
                    default=False,
                    action="store_true",
                    help="whether use a split backbone")
parser.add_argument("--co_view",
                    default=False,
                    action="store_true",
                    help="only use the collaborative view")
parser.add_argument("--se_view",
                    default=False,
                    action="store_true",
                    help="only use the semantic view")


# Other parameters
parser.add_argument("--train_batch_size",
                    default=512,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--lr",
                    default=0.001,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2",
                    default=0,
                    type=float,
                    help='The L2 regularization')
parser.add_argument("--num_train_epochs",
                    default=100,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--lr_dc_step",
                    default=1000,
                    type=int,
                    help='every n step, decrease the lr')
parser.add_argument("--lr_dc",
                    default=0,
                    type=float,
                    help='how many learning rate to decrease')
parser.add_argument("--patience",
                    type=int,
                    default=20,
                    help='How many steps to tolerate the performance decrease while training')
parser.add_argument("--watch_metric",
                    type=str,
                    default='NDCG@10',
                    help="which metric is used to select model.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for different data split")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--gpu_id',
                    default=0,
                    type=int,
                    help='The device id.')
parser.add_argument('--num_workers',
                    default=0,
                    type=int,
                    help='The number of workers in dataloader')
parser.add_argument("--log", 
                    default=False,
                    action="store_true",
                    help="whether create a new log file")

torch.autograd.set_detect_anomaly(True)

args = parser.parse_args()
set_seed(args.seed) # fix the random seed
args.output_dir = os.path.join(args.output_dir, args.dataset)
args.pretrain_dir = os.path.join(args.output_dir, args.pretrain_dir)
args.output_dir = os.path.join(args.output_dir, args.model_name)
args.keepon_path = os.path.join(args.output_dir, args.keepon_path)
args.output_dir = os.path.join(args.output_dir, args.check_path)    # if check_path is none, then without check_path


def main():

    log_manager = Logger(args)  # initialize the log manager
    logger, writer = log_manager.get_logger()    # get the logger
    args.now_str = log_manager.get_now_str()

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")


    os.makedirs(args.output_dir, exist_ok=True)

    # generator is used to manage dataset
    if args.model_name in ['llmesr_gru4rec']:
        generator = GeneratorAllUser(args, logger, device)
    elif args.model_name in ["llmesr_bert4rec"]:
        generator = BertGeneratorAllUser(args, logger, device)
    elif args.model_name in ["llmesr_sasrec"]:
        generator = Seq2SeqGeneratorAllUser(args, logger, device)
    else:
        raise ValueError

    trainer = SeqTrainer(args, logger, writer, device, generator)

    if args.do_test:
        trainer.test()
    elif args.do_emb:
        trainer.save_user_emb()
    elif args.do_group:
        trainer.test_group()
    else:
        trainer.train()

    log_manager.end_log()   # delete the logger threads



if __name__ == "__main__":

    main()



