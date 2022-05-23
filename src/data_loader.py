import os
import logging
from utils.constant import *
from utils.tools import *
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict


class DataReader(object):
    """
    load dataset and preprocess
    Preprocessed data has four columns: uid, iid, label, history
    """
    def __init__(self, path, dataset, label='label', sep='\t', seq_sep=','):
        self.dataset = dataset
        self.path = os.path.join(path, dataset)

        self.train_file = os.path.join(self.path, dataset + TRAIN_SUFFIX)
        self.validation_file = os.path.join(self.path, dataset + VALIDATION_SUFFIX)
        self.test_file = os.path.join(self.path, dataset + TEST_SUFFIX)

        self.sep, self.seq_sep = sep, seq_sep
        self.label = label

        self.train_df, self.validation_df, self.test_df = None, None, None
        self._load_data()

    def _load_data(self):
        """
        Load train/validation/test files
        """
        if os.path.exists(self.train_file):
            logging.info("load train tsv...")
            self.train_df = pd.read_csv(self.train_file, sep=self.sep)
            self.train_df['sample_id'] = self.train_df.index + 1
            self.train_df = self.train_df[[UID, IID, LABEL, SAMPLE_ID, SEQ]]
            logging.info("size of train: %d" % len(self.train_df))
        else:
            raise FileNotFoundError('train file is not found.')

        if os.path.exists(self.validation_file):
            logging.info("load validation tsv...")
            self.validation_df = pd.read_csv(self.validation_file, sep=self.sep)
            self.validation_df['sample_id'] = self.validation_df.index + 1
            self.validation_df = self.validation_df[[UID, IID, LABEL, SAMPLE_ID, SEQ, NEG_ITEM]]
            logging.info("size of validation: %d" % len(self.validation_df))
        else:
            raise FileNotFoundError('validation file is not found.')

        if os.path.exists(self.test_file):
            logging.info("load test tsv...")
            self.test_df = pd.read_csv(self.test_file, sep=self.sep)
            self.test_df['sample_id'] = self.test_df.index + 1
            self.test_df = self.test_df[[UID, IID, LABEL, SAMPLE_ID, SEQ, NEG_ITEM]]
            logging.info("size of test: %d" % len(self.test_df))
        else:
            raise FileNotFoundError('test file is not found.')


class RecDataReader(DataReader):
    def __init__(self, path, dataset_name, label='label', sep='\t', seq_sep=','):
        super().__init__(path, dataset_name, label, sep, seq_sep)
        self.all_df = pd.concat([self.train_df, self.validation_df, self.test_df], ignore_index=True)
        self.user_ids_set = set(self.all_df[UID].tolist())
        self.item_ids_set = set(self.all_df[IID].tolist())
        self.num_nodes = len(self.user_ids_set) + len(self.item_ids_set)
        self.num_item = self.all_df[IID].max()
        self.num_user = len(self.user_ids_set)

        self.all_user2items_dict = self._prepare_user2items_dict(self.all_df)
        self.train_user2items_dict = self._prepare_user2items_dict(self.train_df)
        self.valid_user2items_dict = self._prepare_user2items_dict(self.validation_df)
        self.test_user2items_dict = self._prepare_user2items_dict(self.test_df)

    @staticmethod
    def _prepare_user2items_dict(df):
        df_groups = df.groupby(UID)
        user_sample_dict = defaultdict(set)
        for uid, group in df_groups:
            user_sample_dict[uid] = set(group[IID].tolist())
        return user_sample_dict


class ControllerDataset(Dataset):
    def __init__(self, data_reader, batch_size, stage):
        self.data_reader = data_reader
        self.batch_size = batch_size
        self.stage = stage
        self.data = self._get_data()

    def _get_data(self):
        """
        Convert dataframe into numpy array.
        Columns: 0: uid, 1: iid, 2: label, 3 to end: sequence
        :return: data in numpy format
        """
        if self.stage == 'train':
            df = self.data_reader.train_df[[UID, IID, LABEL, SAMPLE_ID, SEQ]]
        elif self.stage == 'valid':
            df = self.data_reader.validation_df[[UID, IID, LABEL, SAMPLE_ID, SEQ]]
        else:
            df = self.data_reader.test_df[[UID, IID, LABEL, SAMPLE_ID, SEQ]]

        data = df.to_numpy()
        sequence = data[:, 4].tolist()
        sequence = np.asarray([list(map(int, s.split(','))) for s in sequence])
        data = np.delete(data, 4, 1)
        data = np.append(data, sequence, 1)

        return data

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(batch):
        batch = np.asarray(batch).astype(int)
        feed_dict = dict()
        feed_dict['batch_size'] = len(batch)
        feed_dict[UID] = batch[:, 0]
        feed_dict[IID] = batch[:, 1]
        feed_dict[LABEL] = batch[:, 2]
        feed_dict[SAMPLE_ID] = batch[:, 3]
        feed_dict[SEQ] = batch[:, 4:]

        return feed_dict


class ChildDataset:
    def __init__(self, data_reader, stage, batch_size=128, num_neg=-1, shuffle=False):
        self.data_reader = data_reader
        self.batch_size = batch_size
        self.stage = stage
        self.num_neg = num_neg
        self.shuffle = shuffle

        self.batches = None
        self.df = None

    def __len__(self):
        assert self.df is not None
        if self.batches is None:
            logging.info('Generating {} data...'.format(self.stage))
            self.batches = self._prepare_batches(self.df, self.batch_size)
        return len(self.batches) if self.batches else 0

    def __getitem__(self, idx):
        assert self.df is not None
        if self.batches is None:
            logging.info('Generating {} data...'.format(self.stage))
            self.batches = self._prepare_batches(self.df, self.batch_size)
        return self.batches[idx]

    def init(self, arc_df):
        """
        Initialize data batches
        :param arc_df: sampled architecture dataframe
        :return: N/A
        """
        if self.stage == 'train':
            print('\nPrepare Train Data...')
            self.df = self.data_reader.train_df.merge(arc_df, how='right', on=SAMPLE_ID)
        elif self.stage == 'valid':
            self.df = self.data_reader.validation_df.merge(arc_df, how='right', on=SAMPLE_ID)
        elif self.stage == 'test':
            self.df = self.data_reader.test_df.merge(arc_df, how='right', on=SAMPLE_ID)
        else:
            raise ValueError('stage must in ["train", "valid", "test"]')
        self.batches = self._prepare_batches(self.df, self.batch_size)

    def collate_fn(self, batch):
        if self.stage == 'train':
            self.num_neg = 1
        batch['target_head'], batch['target_tail'] = self._get_rec_neg_samples(self.num_neg, batch)
        feed_dict = self._format_df_to_dict(batch)
        feed_dict['batch_size'] = len(batch)
        return feed_dict

    @staticmethod
    def _prepare_batches(data_df, batch_size):
        """
        prepare batches. The first half of each batch are positive samples, the second half
        are negative samples. This method is compatible with 1:n pos to neg sample design.
        :param data_df: input dataframe. Generated from _convert_df_to_expression() method
        :param batch_size: size of each batch
        :return: batches list
        """
        # group by sequence
        sample_seq_groups = data_df.groupby(ARC)
        batches = []
        for _, group_df in sample_seq_groups:
            tmp_num_example = len(group_df)
            tmp_total_batch = int((tmp_num_example + batch_size - 1) / batch_size)
            for batch in range(tmp_total_batch):
                batch_start = batch * batch_size
                batch_end = min(tmp_num_example, batch_start + batch_size)
                real_batch_size = batch_end - batch_start
                batches.append(group_df.iloc[batch_start:batch_start + real_batch_size, :])
        return batches

    @staticmethod
    def _format_df_to_dict(df):
        """
        convert the pandas dataframe into a dictionary where keys are the columns names,
        values are the column contents in the lists
        :param df: dataframe
        :return: dictionary with data info
        """
        data_columns = list(df.columns)
        data_dict = {}
        for c in data_columns:
            data_dict[c] = df[c].tolist()
        return data_dict

    def _get_rec_neg_samples(self, num_neg_sample, df):
        # generate negative samples
        target_head_list = []
        target_tail_list = []
        for row in df.itertuples():
            target_tail = [str(row.iid)]
            if NEG_ITEM in df:
                neg_samples = eval(row.neg_items)
            elif num_neg_sample == -1:
                neg_samples = self._generate_rec_neg_from_all_sample(int(row.uid))
            else:
                neg_samples = self._generate_rec_neg_samples(int(row.uid), num_neg_sample)
            tails = [str(tail) for tail in neg_samples]
            target_tail.extend(tails)
            target_head = [str(row.uid)] * (len(neg_samples) + 1)
            target_head_list.append(','.join(target_head))
            target_tail_list.append(','.join(target_tail))

        return target_head_list, target_tail_list

    def _generate_rec_neg_from_all_sample(self, target_head):
        neg_candidates = list(self.data_reader.item_ids_set - self.data_reader.all_user2items_dict[target_head])
        return neg_candidates

    def _generate_rec_neg_samples(self, target_head, num_neg):
        """
        Generate negative samples for the given expression
        :param num_neg: number of negative samples
        :return: negative samples lists
        """
        neg_candidates = self.data_reader.item_ids_set - self.data_reader.all_user2items_dict[target_head]
        neg_candidates = \
            np.random.choice(list(neg_candidates), num_neg, replace=False).tolist()
        return neg_candidates
