import os
import logging
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import *
from sklearn.metrics import *
from utils.metrics import *


class LogicNetwork(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_item,
                 num_layers,
                 r_weight,
                 dropout=0.0,
                 model_path='../model/child_model.pt',
                 fixed_arc=None
                 ):
        super(LogicNetwork, self).__init__()
        self.dropout = dropout
        self.num_item = num_item
        self.embed_dim = embed_dim
        self.r_weight = r_weight
        self.model_path = model_path
        self.num_layers = num_layers        # number of linear layers for MLP
        self.item_embed = nn.Embedding(self.num_item, embed_dim)

        self.NOT = self._get_ln_module(
            self.embed_dim, self.embed_dim, self.num_layers, self.dropout, batch_norm=False)
        self.AND = self._get_ln_module(
            2 * self.embed_dim, self.embed_dim, self.num_layers, self.dropout, batch_norm=False)
        self.OR = self._get_ln_module(
            2 * self.embed_dim, self.embed_dim, self.num_layers, self.dropout, batch_norm=False)

        self.true = self._init_true_anchor_vector()

        self.encoder = self._get_ln_module(
            self.embed_dim, self.embed_dim, self.num_layers, self.dropout, batch_norm=False)

        self.module_dict = nn.ModuleDict({'AND': self.AND, 'OR': self.OR, 'NOT': self.NOT})

        self.cos_amplify_factor = 10
        self.optimizer = None

    @staticmethod
    def init_params(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def _init_true_anchor_vector(self):
        true = torch.empty(self.embed_dim).unsqueeze(dim=0)
        nn.init.uniform_(true, a=0, b=0.1)
        true = F.normalize(true)
        true = nn.Parameter(true, requires_grad=False)
        return true

    @staticmethod
    def _get_ln_module(input_dim, output_dim, num_layers, dropout, batch_norm=False, is_ic_layer=True):
        """
        Initialize a MLP
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param num_layers: number of layers
        :param batch_norm: if do batch normalization -> BN after weight layer before activation
        :param is_ic_layer: if use IC (independent component) layer -> BN+Dropout after activation
        :return: nn.Sequential
        """
        module_list = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        for i in range(1, num_layers):
            if batch_norm and not is_ic_layer:
                module_list.append(nn.BatchNorm1d(output_dim))  # maybe useless in one-by-one for loop approach?
            module_list.append(nn.LeakyReLU())
            if is_ic_layer:
                if batch_norm:
                    module_list.append(nn.BatchNorm1d(output_dim))
                module_list.append(nn.Dropout(p=dropout))
            module_list.append(nn.Linear(output_dim, output_dim))
        encoder = nn.Sequential(*module_list)
        return encoder

    def predict(self, feed_dict):
        """
        compute forward nn prediction results without calculating loss and gradient.
        Can be called by forward() function for forward nn gradient computation or
        for evaluation only.
        :param: input data dictionary with keys: neighbors, target, sample_index, label and sequence
        :return: prediction
        """
        # prepare input sample_arc: convert from a list of strings into arc dict where key=layer_id, value=arc list
        sample_arc = feed_dict[ARC]
        assert len(set(sample_arc)) == 1
        sample_arc = sample_arc[0].split(';')
        sample_arc_dict = {key: arc_str.split(',') for key, arc_str in enumerate(sample_arc)}

        # prepare sequence: convert from a list of strings into 2D numpy array
        sequences = np.array([x.split(',') for x in feed_dict[SEQ]], dtype=int)

        constraints = []       # for logic constraint use
        intermediates = {}

        # encode sequences
        encoded_seq = self._encode_seq(sequences, sample_arc_dict, intermediates, constraints)

        if self.training:
            encoded_embed, labels, sample_ids = \
                self._encode_target_training(feed_dict, encoded_seq, constraints)
            constraints = torch.cat(constraints)
        else:
            encoded_embed, labels, sample_ids = self._encode_target_evaluate(feed_dict, encoded_seq)

        expanded_true = self.true.expand_as(encoded_embed)
        prediction = F.cosine_similarity(encoded_embed, expanded_true, dim=-1) * self.cos_amplify_factor

        labels = np.array(labels)
        sample_ids = np.array(sample_ids)
        out_dict = {'prediction': prediction,
                    'labels': labels,
                    'sample_id': sample_ids,
                    'check': list()}
        if self.training:
            out_dict['constraints'] = constraints
        return out_dict

    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)
        batch_size = feed_dict['batch_size']
        constraints = out_dict['constraints']
        logic_reg_loss = self.logic_regularization(constraints)

        pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
        # pos, neg = self._group_output(out_dict['prediction'], feed_dict['batch_size'])
        loss = -(pos - neg).sigmoid().log().sum()
        loss = loss + logic_reg_loss
        out_dict['loss'] = loss
        out_dict['check'].append(('logic_loss', logic_reg_loss.detach()))

        return out_dict

    def _arc_parser(self, sample_arc, layer_id, sequence, intermediates):
        layer_id = layer_id
        module_name = sample_arc[layer_id][0]
        seq_len = sequence.shape[-1]

        if sample_arc[layer_id][1].startswith(NEG_SYMBOL):
            input_1_pos = int(sample_arc[layer_id][1][1:])
            if input_1_pos < seq_len:
                input_1_idx = numpy_to_torch(sequence[:, input_1_pos])  # assume the sequence is a numpy array
                input_1_embed = self.item_embed(input_1_idx - 1)        # idx start from 1 not 0
            else:
                input_1_embed = intermediates[input_1_pos]
            input_1 = self.module_dict['NOT'](input_1_embed)
        else:
            input_1_pos = int(sample_arc[layer_id][1])
            if input_1_pos < seq_len:
                input_1_idx = numpy_to_torch(sequence[:, input_1_pos])  # assume the sequence is a numpy array
                input_1_embed = self.item_embed(input_1_idx - 1)        # idx start from 1 not 0
            else:
                input_1_embed = intermediates[input_1_pos]
            input_1 = input_1_embed

        if sample_arc[layer_id][2].startswith(NEG_SYMBOL):
            input_2_pos = int(sample_arc[layer_id][2][1:])
            if input_2_pos < seq_len:
                input_2_idx = numpy_to_torch(sequence[:, input_2_pos])  # assume the sequence is a numpy array
                input_2_embed = self.item_embed(input_2_idx - 1)        # idx start from 1 not 0
            else:
                input_2_embed = intermediates[input_2_pos]
            input_2 = self.module_dict['NOT'](input_2_embed)
        else:
            input_2_pos = int(sample_arc[layer_id][2])
            if input_2_pos < seq_len:
                input_2_idx = numpy_to_torch(sequence[:, input_2_pos])  # assume the sequence is a numpy array
                input_2_embed = self.item_embed(input_2_idx - 1)        # idx start from 1 not 0
            else:
                input_2_embed = intermediates[input_2_pos]
            input_2 = input_2_embed

        return module_name, input_1, input_2

    def _encode_seq(self,
                    sequences,
                    sample_arc_dict,
                    intermediates,
                    constraints,
                    ):
        # this implementation assumes that the data in current feed_dict with exactly the same length
        # fixme: if sequence is empty, then will get error.
        seq_len = sequences.shape[-1]
        output = None
        for layer_id in range(len(sample_arc_dict)):
            module_name, input_1, input_2 = self._arc_parser(sample_arc_dict, layer_id, sequences, intermediates)
            output = self.module_dict[module_name](torch.cat((input_1, input_2), dim=-1))
            intermediates[seq_len + layer_id] = output
            constraints.append(input_1)
            constraints.append(input_2)
            constraints.append(output)
        assert output is not None
        return output

    def _encode_target_training(self, feed_dict, encoded_embed, constraints):
        """
        for training use. Encode the encoded seqs embedding with the target item embedding.
        Only consider 1:1 neg sample case for now.
        :param feed_dict:
        :param encoded_embed:
        :param intermediate_vectors:
        :return:
        """
        # Encode target
        pos_target = []
        neg_target = []
        sample_id_list = []
        for i, target_str in enumerate(feed_dict['target_tail']):
            iid_list = [int(t) for t in target_str.split(',')]
            pos_target.append(iid_list[0])
            neg_target.append(iid_list[1])
        pos_target = numpy_to_torch(np.asarray(pos_target))
        neg_target = numpy_to_torch(np.asarray(neg_target))

        # obtain target item embeddings: pos and neg
        pos_target_embed = self.item_embed(pos_target - 1)
        neg_target_embed = self.item_embed(neg_target - 1)
        constraints.append(pos_target_embed)
        constraints.append(neg_target_embed)

        encoded_pos = self._compute_logic_or(self.module_dict['NOT'](encoded_embed), pos_target_embed)
        encoded_neg = self._compute_logic_or(self.module_dict['NOT'](encoded_embed), neg_target_embed)
        constraints.append(encoded_pos)
        constraints.append(encoded_neg)

        # first half are positive embeddings and the second half for negative samples
        encoded_embed = torch.cat((encoded_pos, encoded_neg), dim=0)
        label_list = [1] * len(feed_dict[SAMPLE_ID]) + [0] * len(feed_dict[SAMPLE_ID])
        sample_id_list.extend(feed_dict[SAMPLE_ID])
        sample_id_list.extend(feed_dict[SAMPLE_ID])

        return encoded_embed, label_list, sample_id_list

    def _encode_target_evaluate(self, feed_dict, encoded_embed):
        label_list = []
        sample_id_list = []
        output = None
        for i, iid_str in enumerate(feed_dict['target_tail']):
            iid_idx = numpy_to_torch(np.asarray([int(t) for t in iid_str.split(',')]))
            target_embed = self.item_embed(iid_idx - 1)

            tmp_labels = [0] * len(iid_idx)
            tmp_labels[0] = 1
            sample_id = [feed_dict[SAMPLE_ID][i]] * len(iid_idx)
            sample_id_list.extend(sample_id)
            label_list.extend(tmp_labels)

            encoded_targets = self._compute_logic_or(
                self.module_dict['NOT'](encoded_embed[i]).expand_as(target_embed), target_embed)
            if output is None:
                output = encoded_targets
            else:
                output = torch.cat((output, encoded_targets), dim=0)

        return output, label_list, sample_id_list

    def logic_regularization(self, vectors):
        dim = len(vectors.size()) - 1
        one = torch.tensor(1.)
        if torch.cuda.device_count() > 0:
            one = one.cuda()
        # length constraint
        # r_length = constraint.norm(dim=dim)()
        false = self.module_dict['NOT'](self.true).view(1, -1)

        # not
        r_not_not_true = \
            (one - F.cosine_similarity(self.module_dict['NOT'](self.module_dict['NOT'](self.true)), self.true, dim=0)).sum()
        r_not_not_self = \
            (one - F.cosine_similarity(self.module_dict['NOT'](self.module_dict['NOT'](vectors)), vectors, dim=dim)).mean()
        r_not_self = (one + F.cosine_similarity(self.module_dict['NOT'](vectors), vectors, dim=dim)).mean()

        # todo: check if need to add inverse to all OR logic regularizers
        # or
        r_or_true = (one - F.cosine_similarity(
            self._compute_logic_or(vectors, self.true.expand_as(vectors)), self.true.expand_as(vectors), dim=dim)) \
            .mean()
        r_or_false = (one - F.cosine_similarity(
            self._compute_logic_or(vectors, false.expand_as(vectors)), vectors, dim=dim)).mean()
        r_or_self = (one - F.cosine_similarity(self._compute_logic_or(vectors, vectors), vectors, dim=dim)).mean()

        r_or_not_self = (one - F.cosine_similarity(
            self._compute_logic_or(vectors, self.module_dict['NOT'](vectors)), self.true.expand_as(vectors), dim=dim)).mean()
        r_or_not_self_inverse = (one - F.cosine_similarity(
            self._compute_logic_or(self.module_dict['NOT'](vectors), vectors), self.true.expand_as(vectors), dim=dim)).mean()
        # True/False
        true_false = one + F.cosine_similarity(self.true.view(-1), false.view(-1), dim=-1)
        r_loss = \
            r_not_not_true + r_not_not_self + r_not_self + true_false + \
            r_or_true + r_or_false + r_or_self + r_or_not_self + r_or_not_self_inverse
        r_loss = r_loss * self.r_weight

        return r_loss

    def _compute_logic_or(self, v1, v2):
        assert len(v1.size()) == len(v2.size())
        input_data = torch.cat((v1, v2), dim=-1)
        output = self.module_dict['OR'](input_data)
        return output

    def _compute_logic_and(self, v1, v2):
        assert len(v1.size()) == len(v2.size())
        input_data = torch.cat((v1, v2), dim=-1)
        output = self.module_dict['AND'](input_data)
        return output

    def save_model(self, model_path=None):
        """
        save model
        """
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        logging.info('Save model to ' + model_path)

    def load_model(self, model_path=None):
        """
        load model
        """
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        logging.info('Load model from ' + model_path)

    def count_variables(self):
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    @staticmethod
    def evaluate_method(p, data, metrics):
        label = data[LABEL]
        evaluations = []
        for metric in metrics:
            if metric == 'rmse':
                evaluations.append(np.sqrt(mean_squared_error(label, p)))
            elif metric == 'mae':
                evaluations.append(mean_absolute_error(label, p))
            elif metric == 'auc':
                evaluations.append(roc_auc_score(label, p))
            elif metric == 'f1':
                evaluations.append(f1_score(label, p))
            elif metric == 'accuracy':
                evaluations.append(accuracy_score(label, p))
            elif metric == 'precision':
                evaluations.append(precision_score(label, p))
            elif metric == 'recall':
                evaluations.append(recall_score(label, p))
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
                    evaluations.append(np.average(ndcgs))
                elif metric.startswith('hit@'):
                    hits = []
                    for uid, group in df_group:
                        hits.append(int(np.sum(group['l'][:k]) > 0))
                    evaluations.append(np.average(hits))
                elif metric.startswith('precision@'):
                    precisions = []
                    for uid, group in df_group:
                        precisions.append(precision_at_k(group['l'].tolist()[:k], k=k))
                    evaluations.append(np.average(precisions))
                elif metric.startswith('recall@'):
                    recalls = []
                    for uid, group in df_group:
                        recalls.append(1.0 * np.sum(group['l'][:k]) / np.sum(group['l']))
                    evaluations.append(np.average(recalls))
                elif metric.startswith('f1@'):
                    f1 = []
                    for uid, group in df_group:
                        num_overlap = 1.0 * np.sum(group['l'][:k])
                        f1.append(2 * num_overlap / (k + 1.0 * np.sum(group['l'])))
                    evaluations.append(np.average(f1))
        return evaluations

    def l2(self):
        """
        calculate l2 regularization
        :return:
        """
        l2 = 0
        for p in self.parameters():
            l2 += (p ** 2).sum()
        return l2
