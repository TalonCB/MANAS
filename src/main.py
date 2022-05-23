import config
import logging
import os
import sys
from torch.utils.data import DataLoader
from utils.tools import *
from data_loader import RecDataReader
from data_loader import ControllerDataset
from data_loader import ChildDataset
from controller import Controller
from controller import ControllerNoneSample
from runner import BaseRunner
from logic_module import LogicNetwork


def main(args):
    # log, result path
    log_file_name = ['ELNAS', args.dataset, str(args.random_seed),
                     'optimizer=' + args.optimizer, 'lr=' + str(args.lr), 'l2=' + str(args.l2),
                     'dropout=' + str(args.dropout), 'batch_size=' + str(args.batch_size)]
    log_file_name = '__'.join(log_file_name).replace(' ', '__')
    if args.log_file == '../log/log.txt':
        args.log_file = '../log/%s.txt' % log_file_name

    # logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # convert the namespace into dictionary e.g. init_args.model_name -> {'model_name': BaseModel}
    logging.info(vars(args))

    # random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info("# cuda devices: %d" % torch.cuda.device_count())

    # load data
    data_reader = RecDataReader(args.data_path, args.dataset, label=args.label, sep=args.sep, seq_sep=args.seq_sep)

    # prepare controller loaders
    controller_data_loader = {}
    for stage in ['train', 'valid', 'test']:
        batch_size = args.batch_size if stage == 'valid' else args.eval_batch_size
        dataset = ControllerDataset(data_reader, batch_size=batch_size, stage=stage)
        shuffle = True if stage == 'train' else False
        controller_data_loader[stage] = \
            DataLoader(dataset=dataset,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       pin_memory=True,
                       num_workers=args.num_workers,
                       collate_fn=ControllerDataset.collate_fn
                       )

    # prepare child network datasets
    child_data_sets = {}
    for stage in ['train', 'valid', 'test']:
        if stage == 'train':
            batch_size = args.batch_size
        elif stage == 'valid':
            batch_size = args.batch_size
        else:
            batch_size = args.eval_batch_size
        # batch_size = args.batch_size if stage == 'train' else args.eval_batch_size
        shuffle = True if stage == 'train' else False
        dataset = ChildDataset(data_reader,
                               batch_size=batch_size,
                               stage=stage,
                               num_neg=args.child_vt_num_neg,
                               shuffle=shuffle)
        child_data_sets[stage] = dataset

    # create controller
    if not args.non_sample:
        controller = Controller(
            num_item=data_reader.num_item,
            num_branches=args.child_num_branches,
            max_num_layers=args.child_max_num_layers,
            lstm_size=args.controller_lstm_size,
            lstm_num_layers=args.controller_lstm_num_layers,
            tanh_constant=args.controller_tanh_constant,
            temperature=None,
            model_path=args.controller_model_path
        )
    else:
        controller = ControllerNoneSample(
            num_item=data_reader.num_item,
            num_branches=args.child_num_branches,
            max_num_layers=args.child_max_num_layers,
            lstm_size=args.controller_lstm_size,
            lstm_num_layers=args.controller_lstm_num_layers,
            tanh_constant=args.controller_tanh_constant,
            temperature=None,
            model_path=args.controller_model_path
        )

    controller.apply(Controller.init_params)

    # create shared logic module
    shared_logic_module = LogicNetwork(embed_dim=args.child_embed_size,
                                       num_item=data_reader.num_item,
                                       num_layers=2,
                                       r_weight=args.child_logic_reg_weight,
                                       model_path=args.child_model_path,
                                       )
    shared_logic_module.apply(LogicNetwork.init_params)

    if args.resume or args.test_only:
        controller.load_model()
        shared_logic_module.load_model()

    if torch.cuda.device_count() > 0:
        controller = controller.cuda()
        shared_logic_module = shared_logic_module.cuda()

    # create runner
    runner = BaseRunner(metrics=args.metric,
                        optimizer=args.optimizer,
                        controller_lr=args.controller_lr,
                        controller_train_step=args.controller_train_step,
                        child_l2=args.child_l2_reg,
                        check_epoch=args.check_epoch)

    logging.info('Test Before Training = ' + format_metric(
        runner.eval_model(controller, shared_logic_module, controller_data_loader, child_data_sets, 'test', args))
                 + ' ' + ','.join(runner.metrics))

    if not args.test_only:
        runner.train(controller, shared_logic_module, controller_data_loader, child_data_sets, args)

    best, worst, average, std = runner.eval_sample_models(
        controller, shared_logic_module, controller_data_loader, child_data_sets, 'test', args,
        export_arch=args.export_arch, arch_file_name=args.arch_file_name
    )
    logging.info('Test After Training Best = ' + format_metric(best) + ' ' + ','.join(runner.metrics))
    logging.info('Test After Training Worst = ' + format_metric(worst) + ' ' + ','.join(runner.metrics))
    logging.info('Test After Training Average = ' + format_metric(average) + ' ' + ','.join(runner.metrics))
    logging.info('Test After Training STD = ' + format_metric(std) + ' ' + ','.join(runner.metrics))


if __name__ == "__main__":
    args, unknown_args = config.parse_args()
    main(args)
