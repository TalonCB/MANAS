import gc
import os
import logging
from sklearn.metrics import *
from utils.metrics import *
from collections import defaultdict
from time import time
from tqdm import tqdm
from utils.tools import *
from utils.constant import *
from torch.utils.data import DataLoader


class BaseRunner:
    def __init__(self,
                 metrics,
                 optimizer='Adam',
                 controller_lr=0.001,
                 controller_train_step=50,
                 eval_batch_size=128 * 128,
                 child_l2=1e-5,
                 start_epoch=1,
                 check_epoch=10,
                 early_stop=1,
                 num_worker=1):
        self.optimizer = optimizer
        self.controller_lr = controller_lr
        self.controller_train_step = controller_train_step
        self.eval_batch_size = eval_batch_size
        self.child_l2 = child_l2
        self.time = None
        self.valid_results, self.test_results = [], []
        self.metrics = metrics.lower().split(',')

    def train(self, controller, shared_logic_module, controller_data_loaders, child_datasets, args):
        baseline = None
        valid_set = 'valid'
        test_set = 'test'
        for epoch in range(1, args.epoch + 1):
            self._check_time()
            if epoch < args.epoch:
                child_train_mode = 'partial'
            else:
                child_train_mode = 'full'

            # Train child logic network
            self.train_child(
                child_train_mode,
                controller,
                shared_logic_module,
                controller_data_loaders,
                child_datasets,
                args
            )

            # if not the last epoch, train the controller
            if epoch < args.epoch:
                baseline = self.train_controller(
                    epoch,
                    controller,
                    shared_logic_module,
                    controller_data_loaders,
                    child_datasets,
                    args,
                    baseline
                )

            training_time = self._check_time()
            if epoch % args.eval_every_epoch == 0:
                valid_result = self.eval_model(
                    controller,
                    shared_logic_module,
                    controller_data_loaders,
                    child_datasets,
                    valid_set,
                    args
                )
                test_result = self.eval_model(
                    controller,
                    shared_logic_module,
                    controller_data_loaders,
                    child_datasets,
                    test_set,
                    args
                )

                self.valid_results.append(valid_result)
                self.test_results.append(test_result)
                testing_time = self._check_time()
                logging.info("Epoch %5d [%.1f s]\t validation= %s test= %s [%.1f s] "
                             % (epoch + 1, training_time, format_metric(valid_result),
                                format_metric(test_result), testing_time) + ','.join(self.metrics))

                if best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
                    controller.save_model()
                    shared_logic_module.save_model()
            else:
                logging.info("Epoch %5d [%.1f s]" % (epoch + 1, training_time))

    @staticmethod
    def fit(model, batches, epoch=-1, num_batches=0):  # fit the results for an input set
        """
        Train the model
        :param model: model instance
        :param batches: train data in batches
        :param num_batches: number of batches
        :param epoch: epoch number
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()
        model.train()
        loss_list = list()
        output_dict = dict()
        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
                          ncols=100, mininterval=1, total=num_batches):
            batch = batch_to_gpu(batch)
            model.optimizer.zero_grad()
            result_dict = model(batch)
            loss = result_dict['loss']
            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), 50)
            model.optimizer.step()
            loss_list.append(result_dict['loss'].detach().cpu().data.numpy())
            # if batch is batches[-1]:
            output_dict['check'] = result_dict['check']
        output_dict['loss'] = np.mean(loss_list)
        return output_dict

    def train_child(self, train_mode, controller, shared_logic_module, controller_data_loaders, child_datasets, args):
        child_step = args.child_full_step if train_mode == 'full' else args.child_train_step
        controller_train_dl = controller_data_loaders['train']
        child_train_data = child_datasets['train']

        sample_id_list = []
        arc_string_list = []

        controller.eval()
        shared_logic_module.train()

        # Generate logic networks
        for data in tqdm(controller_train_dl, leave=False, ncols=100, mininterval=1, desc='Child-Network Generation'):
            with torch.no_grad():
                controller(data)
            sample_id_list.extend(data[SAMPLE_ID])
            arc_string_list.extend(controller.seq_string)

        # prepare arc data for child network
        data_df = pd.DataFrame()
        data_df[SAMPLE_ID] = sample_id_list
        data_df[ARC] = arc_string_list

        # create child train dataloader
        child_train_data.init(data_df)
        child_train_dl = DataLoader(dataset=child_train_data,
                                    shuffle=True,
                                    batch_size=None,
                                    num_workers=args.num_workers,
                                    pin_memory=True,
                                    collate_fn=child_train_data.collate_fn)
        # Initialize child optimizer
        if shared_logic_module.optimizer is None:
            shared_logic_module.optimizer = \
                torch.optim.Adam(shared_logic_module.parameters(),
                                 lr=args.child_lr,
                                 weight_decay=args.child_l2_reg)
        # train network
        for step in range(child_step):
            start = time()
            output_dict = self.fit(shared_logic_module,
                                   child_train_dl,
                                   epoch=step,
                                   num_batches=len(child_train_dl))
            if args.check_epoch > 0 and (step == 1 or step % args.check_epoch == 0):
                self.check(shared_logic_module, output_dict)
            end = time()
            logging.info("training time: {:.2f}it/s".format(1. / (end - start)))
        controller.train()

    def eval_model(self, controller, shared_logic_module, controller_data_loaders, child_datasets, dataset_name, args,
                   export_arch=False, arch_file_name=None):
        # todo: need to eval multiple archs and return the best performance
        controller.eval()
        shared_logic_module.eval()

        controller_dl = controller_data_loaders[dataset_name]
        child_valid_data = child_datasets[dataset_name]

        sample_id_list = []
        arc_string_list = []
        # Generate logic networks
        for data in tqdm(controller_dl, leave=False, ncols=100, mininterval=1, desc='Child-Network Generation'):
            start_time = time()
            with torch.no_grad():
                controller(data)
            sample_id_list.extend(data[SAMPLE_ID])
            arc_string_list.extend(controller.seq_string)

        # prepare arc data for child network
        data_df = pd.DataFrame()
        data_df[SAMPLE_ID] = sample_id_list
        data_df[ARC] = arc_string_list

        # create child train dataloader
        child_valid_data.init(data_df)
        if export_arch:
            arch_file_name = args.arch_file_name if arch_file_name is None else arch_file_name
            path = os.path.join(args.arch_path, arch_file_name + '.tsv')
            df = child_valid_data.df
            df.to_csv(path, sep='\t', index=False)

        child_valid_dl = DataLoader(dataset=child_valid_data,
                                    shuffle=True,
                                    batch_size=None,
                                    num_workers=args.num_workers,
                                    pin_memory=True,
                                    collate_fn=child_valid_data.collate_fn)

        shared_logic_module.eval()
        result_dict = defaultdict(list)
        evaluations = []
        metrics = args.metric.split(',')
        for batch in tqdm(child_valid_dl, leave=False, desc='Evaluating child on {} set'.format(dataset_name),
                          ncols=100, mininterval=1, total=len(child_valid_dl)):
            batch = batch_to_gpu(batch)
            out_dict = shared_logic_module.predict(batch)
            prediction = out_dict['prediction']
            labels = out_dict['labels']
            sample_ids = out_dict['sample_id']
            prediction = prediction.detach().cpu().numpy()
            data_dict = {LABEL: labels, SAMPLE_ID: sample_ids}
            results = self.evaluate_method(prediction, data_dict, metrics=metrics)
            for key in results:
                result_dict[key].extend(results[key])
        for metric in metrics:
            evaluations.append(np.average(result_dict[metric]))

        controller.train()
        shared_logic_module.train()
        return evaluations

    def eval_sample_models(self, controller, shared_logic_module, controller_data_loaders, child_datasets,
                           dataset_name, args, export_arch=False, arch_file_name=None):
        results = []
        arch_file_name = args.arch_file_name if arch_file_name is None else arch_file_name
        for i in tqdm(range(args.eval_num_sample), leave=False, desc='Sample Evaluation', ncols=100, mininterval=1):
            tmp_arch_file_name = arch_file_name + '_{}'.format(i)
            result = self.eval_model(controller, shared_logic_module, controller_data_loaders, child_datasets,
                                     dataset_name, args, export_arch, tmp_arch_file_name)
            results.append(result)

        # todo: currently only metrics with the larger the better ones can be used here
        best = max(np.asarray(results), key=sum).tolist()
        worst = min(np.asarray(results), key=sum).tolist()
        average = np.mean(np.asarray(results), axis=0).tolist()
        std = np.std(np.asarray(results), axis=0).tolist()

        return best, worst, average, std

    def train_controller(self,
                         epoch,
                         controller,
                         shared_logic_module,
                         controller_data_loaders,
                         child_datasets,
                         args,
                         baseline=None):
        """Train controller to optimizer validation accuracy using REINFORCE.
        Args:
            epoch: Current epoch.
            controller: Controller module that generates architectures to be trained.
            shared_cnn: CNN that contains all possible architectures, with shared weights.
            data_loaders: Dict containing data loaders.
            controller_optimizer: Optimizer for the controller.
            baseline: The baseline score (i.e. average val_acc) from the previous epoch

        Returns:
            baseline: The baseline score (i.e. average val_acc) for the current epoch
        For more stable training we perform weight updates using the average of
        many gradient estimates. controller_num_aggregate indicates how many samples
        we want to average over (default = 20). By default PyTorch will sum gradients
        each time .backward() is called (as long as an optimizer step is not taken),
        so each iteration we divide the loss by controller_num_aggregate to get the
        average.
        https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L270
        """
        print('Epoch ' + str(epoch) + ': Training controller')
        shared_logic_module.eval()
        controller.train()
        controller_optimizer = torch.optim.Adam(params=controller.parameters(),
                                                lr=args.controller_lr,
                                                betas=(0.0, 0.999),
                                                eps=1e-3)
        child_valid_data = child_datasets['valid']

        reward_meter = AverageMeter()
        baseline_meter = AverageMeter()
        val_metric_meter = AverageMeter()
        loss_meter = AverageMeter()
        train_data = controller_data_loaders['valid']

        for i in range(self.controller_train_step):
            start = time()
            # torch.cuda.empty_cache()
            # gc.collect()
            loss = numpy_to_torch(np.asarray(0.))
            for data in tqdm(train_data, leave=False, ncols=100, mininterval=1, desc='Controller Training'):
                data = batch_to_gpu(data)
                controller(data)

                # prepare arc data for child network
                data_df = pd.DataFrame()
                data_df[SAMPLE_ID] = data[SAMPLE_ID]
                data_df[ARC] = controller.seq_string

                # create child train dataloader
                child_valid_data.init(data_df)
                child_valid_dl = DataLoader(dataset=child_valid_data,
                                            shuffle=True,
                                            batch_size=None,
                                            num_workers=args.num_workers,
                                            pin_memory=True,
                                            collate_fn=child_valid_data.collate_fn)
                result_dict = defaultdict(list)
                evaluations = []
                metrics = args.metric.split(',')
                with torch.no_grad():
                    for batch in tqdm(child_valid_dl, leave=False, desc='Evaluating child on validation set',
                                      ncols=100, mininterval=1, total=len(child_valid_dl)):
                        batch = batch_to_gpu(batch)
                        out_dict = shared_logic_module.predict(batch)
                        prediction = out_dict['prediction'].detach()
                        labels = out_dict['labels']
                        sample_ids = out_dict['sample_id']
                        prediction = prediction.cpu().numpy()
                        data_dict = {LABEL: labels, SAMPLE_ID: sample_ids}
                        results = self.evaluate_method(prediction, data_dict, metrics=metrics)
                        for key in results:
                            result_dict[key].extend(results[key])

                    for metric in metrics:
                        evaluations.append(np.average(result_dict[metric]))

                    val_metric = numpy_to_torch(np.asarray(np.sum(evaluations))).detach()

                reward = val_metric
                reward += args.controller_entropy_weight * controller.sample_entropy

                if baseline is None:
                    baseline = val_metric
                else:
                    baseline = args.controller_bl_dec * baseline + (1 - args.controller_bl_dec) * reward

                loss += -1 * controller.sample_log_prob * (reward - baseline)

                reward_meter.update(reward.detach().item())
                baseline_meter.update(baseline.detach().item())
                val_metric_meter.update(val_metric.detach().item())
                loss_meter.update(loss.detach().item())
                baseline = baseline.detach()

            end = time()
            controller_optimizer.zero_grad()
            loss = loss / len(train_data)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), args.child_grad_bound)
            controller_optimizer.step()

            if i % args.train_controller_print_every == 0:
                learning_rate = controller_optimizer.param_groups[0]['lr']
                display = 'ctrl_step=' + str(i) + \
                          '\tloss=%.3f' % (loss_meter.val) + \
                          '\tent=%.2f' % (controller.sample_entropy.item()) + \
                          '\tlr=%.4f' % (learning_rate) + \
                          '\t|g|=%.4f' % (grad_norm.item()) + \
                          '\tmetric_agg=%.4f' % (val_metric_meter.val) + \
                          '\tbl=%.2f' % (baseline_meter.val) + \
                          '\ttime=%.2fit/s' % (1. / (end - start))
                print(display)

        shared_logic_module.train()
        return baseline

    @staticmethod
    def evaluate_method(p, data, metrics):
        label = data[LABEL]
        evaluations = {}
        for metric in metrics:
            if metric == 'rmse':
                evaluations[metric] = [np.sqrt(mean_squared_error(label, p))]
            elif metric == 'mae':
                evaluations[metric] = [mean_absolute_error(label, p)]
            elif metric == 'mrr':
                df = pd.DataFrame()
                df[SAMPLE_ID] = data[SAMPLE_ID]
                df['p'] = p
                df['l'] = label
                df = df.sort_values(by='p', ascending=False)
                df_group = df.groupby(SAMPLE_ID)
                mrr = []
                for uid, group in df_group:
                    mrr.append(reciprocal_rank(group['l'].tolist()))
                evaluations[metric] = mrr
            else:
                k = int(metric.split('@')[-1])
                df = pd.DataFrame()
                df[SAMPLE_ID] = data[SAMPLE_ID]
                df['p'] = p
                df['l'] = label
                df = df.sort_values(by='p', ascending=False)
                df_group = df.groupby(SAMPLE_ID)
                if metric.startswith('ndcg@'):
                    ndcgs = []
                    for uid, group in df_group:
                        ndcgs.append(ndcg_at_k(group['l'].tolist()[:k], k=k, method=1))
                    evaluations[metric] = ndcgs
                elif metric.startswith('hit@'):
                    hits = []
                    for uid, group in df_group:
                        hits.append(int(np.sum(group['l'][:k]) > 0))
                    evaluations[metric] = hits
                elif metric.startswith('precision@'):
                    precisions = []
                    for uid, group in df_group:
                        precisions.append(precision_at_k(group['l'].tolist()[:k], k=k))
                    evaluations[metric] = precisions
                elif metric.startswith('recall@'):
                    recalls = []
                    for uid, group in df_group:
                        recalls.append(1.0 * np.sum(group['l'][:k]) / np.sum(group['l']))
                    evaluations[metric] = recalls
                elif metric.startswith('f1@'):
                    f1 = []
                    for uid, group in df_group:
                        num_overlap = 1.0 * np.sum(group['l'][:k])
                        f1.append(2 * num_overlap / (k + 1.0 * np.sum(group['l'])))
                    evaluations[metric] = f1
        return evaluations

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def check(self, model, out_dict):
        check = out_dict
        logging.info(os.linesep)
        for i, t in enumerate(check['check']):
            print(t[1])
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss, l2 = check['loss'], model.l2()
        l2 = l2 * self.child_l2
        l2 = l2.detach()
        logging.info('loss = %.4f, l2 = %.4f' % (loss, l2))
        if not (np.absolute(loss) * 0.005 < l2 < np.absolute(loss) * 0.1):
            logging.warning('l2 inappropriate: loss = %.4f, l2 = %.4f' % (loss, l2))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count