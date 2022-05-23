import argparse
import logging


parser = argparse.ArgumentParser(description='MANAS')

# Global args
parser.add_argument('--gpu', type=str, default='2',
                    help='Set CUDA_VISIBLE_DEVICES')
parser.add_argument('--verbose', type=int, default=logging.INFO,
                    help='Logging Level, 0, 10, ..., 50')
parser.add_argument('--log_file', type=str, default='../log/log.txt',
                    help='Logging file path')
# parser.add_argument('--result_file', type=str, default='../result/result.npy',
#                     help='Result file path')
parser.add_argument('--random_seed', type=int, default=2021,
                    help='Random seed of numpy and pytorch.')
parser.add_argument('--resume', action='store_true', help='resume training process.')
parser.add_argument('--test_only', action='store_true', help='test only without training.')
parser.add_argument('--export_arch', action='store_true', help='export archs in testing set')
parser.add_argument('--arch_path', type=str, default='../arch/')
parser.add_argument('--arch_file_name', type=str, default='default_arch')

# Dataset
parser.add_argument('--data_path', type=str, default='../dataset/',
                    help='Input data dir.')
parser.add_argument('--dataset', type=str, default='toy',
                    help='Choose a dataset.')
parser.add_argument('--sep', type=str, default='\t',
                    help='separator of csv file.')
parser.add_argument('--seq_sep', type=str, default=',',
                    help='sequence separator of data file.')
parser.add_argument('--label', type=str, default='label',
                    help='name of dataset label column.')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of process for data processing')

# Runner
parser.add_argument('--train', type=int, default=1,
                    help='To train the model or not.')
parser.add_argument('--load', type=int, default=0,
                    help='Whether load model and continue to train')
parser.add_argument('--epoch', type=int, default=100,
                    help='Number of epochs.')
parser.add_argument('--check_epoch', type=int, default=1,
                    help='Check every epochs.')
parser.add_argument('--early_stop', type=int, default=1,
                    help='whether to early-stop.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size during training.')
parser.add_argument('--eval_batch_size', type=int, default=1024,
                    help='Batch size during testing.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout probability for each deep layer')
parser.add_argument('--l2', type=float, default=1e-4,
                    help='Weight of l2_regularize in loss.')
parser.add_argument('--optimizer', type=str, default='GD',
                    help='optimizer: GD, Adam, Adagrad')
parser.add_argument('--skip_eval', type=int, default=0,
                    help='number of epochs without evaluation')
parser.add_argument('--metric', type=str, default='ndcg@5,ndcg@10,hit@5,hit@10',
                    help='evaluation metrics for child network')
parser.add_argument('--eval_every_epoch', type=int, default=1,
                    help='evaluate child network every n epochs')
parser.add_argument('--train_controller_print_every', type=int, default=10,
                    help='evaluate child network every n epochs')
parser.add_argument('--eval_num_sample', type=int, default=1,
                    help='number of samples for each evaluation')

# Network
parser.add_argument('--non_sample', action='store_true')
parser.add_argument('--controller_lstm_size', type=int, default=64)
parser.add_argument('--controller_lstm_num_layers', type=int, default=1)
parser.add_argument('--controller_entropy_weight', type=float, default=0.0001)
parser.add_argument('--controller_train_every', type=int, default=1)
parser.add_argument('--controller_num_aggregate', type=int, default=20)
parser.add_argument('--controller_train_step', type=int, default=50)
parser.add_argument('--controller_lr', type=float, default=0.005)
parser.add_argument('--controller_tanh_constant', type=float, default=1.5)
parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5)
parser.add_argument('--controller_skip_target', type=float, default=0.4)
parser.add_argument('--controller_skip_weight', type=float, default=0.8)
parser.add_argument('--controller_bl_dec', type=float, default=0.99)
parser.add_argument('--controller_model_path', type=str, default='../model/controller.pt')

parser.add_argument('--child_grad_bound', type=float, default=5.0)
parser.add_argument('--child_l2_reg', type=float, default=1e-5)
parser.add_argument('--child_num_branches', type=int, default=2)
parser.add_argument('--child_train_step', type=int, default=1)
parser.add_argument('--child_full_step', type=int, default=100)
parser.add_argument('--child_max_num_layers', type=int, default=10)
# parser.add_argument('--child_keep_prob', type=float, default=0.9)
parser.add_argument('--child_lr', type=float, default=0.001)
# parser.add_argument('--child_lr_min', type=float, default=0.0005)
# parser.add_argument('--child_lr_T', type=float, default=10)
parser.add_argument('--child_embed_size', type=int, default=64)
parser.add_argument('--child_seq_len', type=int, default=4)
parser.add_argument('--child_vt_num_neg', type=int, default=-1,
                    help='number of negative samples for evaluation')
parser.add_argument('--child_full_epoch', type=int, default=100,
                    help='Max number of epochs to train to converge.')
parser.add_argument('--child_logic_reg_weight', type=float, default=1e-5,
                    help='logic regularization weight')
parser.add_argument('--child_model_path', type=str, default='../model/child_model.pt',
                    help='child model saved path')


def parse_args():
    """
    Parses all of the arguments above
    """
    args, unparsed = parser.parse_known_args()
    return args, unparsed

