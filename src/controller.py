import numpy as np
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils.constant import *
from utils.tools import *
import os
import logging


class Controller(nn.Module):
    """
    Reference to https://github.com/TalonCB/enas_pytorch/blob/master/models/controller.py
    """
    def __init__(self,
                 num_item,
                 num_branches=2,
                 max_num_layers=10,
                 lstm_size=32,
                 lstm_num_layers=2,
                 tanh_constant=2.5,
                 temperature=None,
                 model_path='../model/controller.pt'
                 ):
        super(Controller, self).__init__()
        self.num_item = num_item
        self.num_branches = num_branches
        self.max_num_layers = max_num_layers
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        self.module_id2name = {0: 'AND', 1: 'OR'}
        self.inf = torch.tensor(float('inf'), requires_grad=False)
        self.model_path = model_path

        self._create_params()

    def _create_params(self):
        self.w_lstm = nn.LSTM(input_size=self.lstm_size,
                              hidden_size=self.lstm_size,
                              num_layers=self.lstm_num_layers,
                              batch_first=True)

        self.g_emb = nn.Embedding(1, self.lstm_size)  # Learn the starting input

        self.w_emb = nn.Embedding(self.num_branches, self.lstm_size)
        self.w_soft = nn.Linear(self.lstm_size, self.num_branches, bias=False)

        self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
        self.item_embeddings = nn.Embedding(self.num_item, self.lstm_size)

    @staticmethod
    def init_params(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.uniform_(m.weight, -0.1, 0.1)
            if m.bias is not None:
                torch.nn.init.uniform_(m.bias, -0.1, 0.1)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        elif type(m) == torch.nn.LSTM:
            torch.nn.init.uniform_(m.weight_hh_l0, -0.1, 0.1)
            torch.nn.init.uniform_(m.weight_ih_l0, -0.1, 0.1)

    def forward(self, feed_dict):
        sequence = numpy_to_torch(feed_dict[SEQ]).squeeze(0) - 1
        item_embeds = self.item_embeddings(sequence)
        if len(sequence.size()) < 2:
            item_embeds = item_embeds.unsqueeze(0)
        h0 = None  # setting h0 to None will initialize LSTM state with 0s
          # batch size, seq length, embed dim
        # anchors = [embed for embed in item_embeds]
        anchors = item_embeds
        anchors_w_1 = self.w_attn_1(item_embeds)

        arc_seq = {}
        entropys = []
        log_probs = []
        inp_count = []
        # sample_count = defaultdict(int, {key: 0 for key in range(len(anchors[0]))})   # key: variable index, value: sampled times
        sample_count = np.zeros(shape=[item_embeds.size(0), item_embeds.size(1)])
        inputs = self.g_emb(torch_to_gpu(torch.tensor([0] * item_embeds.size(0))))

        stop = False
        layer_id = 0

        while not stop and layer_id < self.max_num_layers:
            # sample logic module
            inputs = inputs.unsqueeze(1)
            output, hn = self.w_lstm(inputs, h0)
            h0 = hn
            logits = self.w_soft(output)
            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                logits = self.tanh_constant * torch.tanh(logits)
            branch_id_dist = Categorical(logits=logits)
            branch_id = branch_id_dist.sample()
            # branch_name = self.module_id2name[branch_id.item()]
            branch_name = [[self.module_id2name[b_id.item()]] for b_id in branch_id]

            arc_seq[str(layer_id)] = branch_name
            log_prob = branch_id_dist.log_prob(branch_id)
            log_probs.append(log_prob.view(-1))
            entropy = branch_id_dist.entropy()
            entropys.append(entropy.view(-1))

            inputs = self.w_emb(branch_id)
            output, hn = self.w_lstm(inputs, h0)
            # sample input
            # query = torch.reshape(torch.cat(anchors_w_1), shape=(output.size(0), -1, output.size(-1)))
            query = anchors_w_1
            query = torch.tanh(query + self.w_attn_2(output))
            query = self.v_attn(query)
            logits = query.squeeze(-1)

            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                logits = self.tanh_constant * torch.tanh(logits)

            for i in range(logits.size(0)):
                for j in range(logits.size(1)):
                    if sample_count[i][j] > 0:
                        logits[i][j] -= 100

            # inp_dist = Categorical(logits=logits)
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            inp_entropy = -(log_prob * probs).sum(-1, keepdim=False)

            inp = torch.multinomial(probs, 2, replacement=False)
            selected_log_prob = log_prob.gather(1, inp)

            # AND or OR module needs two inputs
            for r in range(len(inp)):
                sample_count[r][inp[r][0].item()] += 1
                sample_count[r][inp[r][1].item()] += 1

            # sample NOT for each input
            not_sample_logits = logits.gather(dim=1, index=inp).unsqueeze(-1)
            # not_sample_logits = torch.cat([not_sample_logits[inp[0]], not_sample_logits[inp[1]]]).unsqueeze(1)
            not_sample_logits = torch.cat([-not_sample_logits, not_sample_logits], dim=-1)
            not_dist = Categorical(logits=not_sample_logits)
            not_sample = not_dist.sample()

            # reorder the inputs to guarantee the variable with a smaller index is at the first place
            ordered = [[0, 1] if row[0].item() < row[1].item() else [1, 0] for row in inp]

            for row, order in enumerate(ordered):
                for i in order:
                    if not_sample[row][i].item() == 1:
                        arc_seq[str(layer_id)][row].append(str(inp[row][i].item()))
                    else:
                        arc_seq[str(layer_id)][row].append(NEG_SYMBOL + str(inp[row][i].item()))

            log_inp_prob = selected_log_prob.view(-1)
            log_inp_prob = torch.sum(log_inp_prob)
            log_probs.append(log_inp_prob.view(-1))

            log_not_prob = not_dist.log_prob(not_sample)
            log_not_prob = torch.sum(log_not_prob)
            log_probs.append(log_not_prob.view(-1))

            entropy_inp = torch.sum(inp_entropy)
            entropy_not = not_dist.entropy()
            entropy_not = torch.sum(entropy_not)
            entropys.append(entropy_inp.view(-1))
            entropys.append(entropy_not.view(-1))

            # Calculate average hidden state of all nodes that got skips
            # and use it as input for next step
            # inp = inp.type(torch.float).squeeze(1)
            one_hot = torch.nn.functional.one_hot(inp, anchors.size(1))
            inp = torch.sum(one_hot, dim=1, keepdim=False).type(torch.float)
            inp_count.append(torch.sum(inp))
            inputs = torch.matmul(inp.unsqueeze(1), anchors).squeeze(1)
            inputs /= (1.0 + torch.sum(inp, dim=-1, keepdim=True))

            layer_id += 1
            anchors = torch.cat((anchors, output), dim=1)
            anchors_w_1 = torch.cat((anchors_w_1, output), dim=1)

            # check stop search condition
            if np.all(sample_count):
                stop = True
            sample_count = np.concatenate((sample_count, np.zeros(shape=[sample_count.shape[0], 1])), axis=1)

        self.sample_arc = arc_seq
        self.seq_string = self._arc_seq_to_string(self.sample_arc)

        entropys = torch.cat(entropys)
        self.sample_entropy = torch.sum(entropys)

        log_probs = torch.cat(log_probs)
        self.sample_log_prob = torch.sum(log_probs)

    @staticmethod
    def _arc_seq_to_string(seq_dict):
        output = None
        for key in range(len(seq_dict)):
            tmp = np.asarray(seq_dict[str(key)])
            tmp = np.array(list(map(lambda x: np.array([','.join(x)]), tmp)))
            output = np.concatenate((output, tmp), axis=1) if output is not None else tmp
        output = np.asarray(list(map(lambda x: ';'.join(x), output)))
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


class ControllerNoneSample(Controller):
    def __init__(self,
                 num_item,
                 num_branches=2,
                 max_num_layers=10,
                 lstm_size=32,
                 lstm_num_layers=2,
                 tanh_constant=2.5,
                 temperature=None,
                 model_path='../model/controller.pt'
                 ):
        super().__init__(num_item, num_branches, max_num_layers, lstm_size, lstm_num_layers,
                         tanh_constant, temperature, model_path)

    def forward(self, feed_dict):
        sequence = numpy_to_torch(feed_dict[SEQ]).squeeze(0) - 1
        item_embeds = self.item_embeddings(sequence)
        if len(sequence.size()) < 2:
            item_embeds = item_embeds.unsqueeze(0)

        h0 = None  # setting h0 to None will initialize LSTM state with 0s
        anchors = item_embeds
        anchors_w_1 = self.w_attn_1(item_embeds)

        arc_seq = {}
        entropys = []
        log_probs = []
        inp_count = []
        sample_count = np.zeros(shape=[item_embeds.size(0), item_embeds.size(1)])
        inputs = self.g_emb(torch_to_gpu(torch.tensor([0] * item_embeds.size(0))))

        stop = False
        layer_id = 0

        while not stop and layer_id < self.max_num_layers:
            # sample logic module
            inputs = inputs.unsqueeze(1)
            output, hn = self.w_lstm(inputs, h0)
            h0 = hn
            logits = self.w_soft(output)
            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                logits = self.tanh_constant * torch.tanh(logits)

            log_prob, branch_id = torch.topk(F.log_softmax(logits, dim=-1), k=1)
            branch_id = branch_id.squeeze(1)
            branch_name = [[self.module_id2name[b_id.item()]] for b_id in branch_id]

            arc_seq[str(layer_id)] = branch_name
            log_probs.append(log_prob.view(-1))

            inputs = self.w_emb(branch_id)
            output, hn = self.w_lstm(inputs, h0)
            # sample input
            query = anchors_w_1
            query = torch.tanh(query + self.w_attn_2(output))
            query = self.v_attn(query)
            logits = query.squeeze(-1)

            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                logits = self.tanh_constant * torch.tanh(logits)

            for i in range(logits.size(0)):
                for j in range(logits.size(1)):
                    if sample_count[i][j] > 0:
                        logits[i][j] -= 100

            log_prob = F.log_softmax(logits, dim=-1)
            selected_log_prob, inp = torch.topk(log_prob, k=2)

            # AND or OR module needs two inputs
            for r in range(len(inp)):
                sample_count[r][inp[r][0].item()] += 1
                sample_count[r][inp[r][1].item()] += 1

            # sample NOT for each input
            # todo(hanxiong): do we still use the same logits or use a different one? Currently use the same one
            not_sample_logits = logits.gather(dim=1, index=inp).unsqueeze(-1)
            not_sample_logits = torch.cat([-not_sample_logits, not_sample_logits], dim=-1)
            not_dist = Categorical(logits=not_sample_logits)
            not_sample = not_dist.sample()

            # reorder the inputs to guarantee the variable with a smaller index is at the first place
            ordered = [[0, 1] if row[0].item() < row[1].item() else [1, 0] for row in inp]

            for row, order in enumerate(ordered):
                for i in order:
                    if not_sample[row][i].item() == 1:
                        arc_seq[str(layer_id)][row].append(str(inp[row][i].item()))
                    else:
                        arc_seq[str(layer_id)][row].append(NEG_SYMBOL + str(inp[row][i].item()))

            log_inp_prob = selected_log_prob.view(-1)
            log_inp_prob = torch.sum(log_inp_prob)
            log_probs.append(log_inp_prob.view(-1))

            log_not_prob = not_dist.log_prob(not_sample)
            log_not_prob = torch.sum(log_not_prob)
            log_probs.append(log_not_prob.view(-1))

            entropy_not = not_dist.entropy()
            entropy_not = torch.sum(entropy_not)
            entropys.append(entropy_not.view(-1))

            # Calculate average hidden state of all nodes that got skips
            # and use it as input for next step
            one_hot = torch.nn.functional.one_hot(inp, anchors.size(1))
            inp = torch.sum(one_hot, dim=1, keepdim=False).type(torch.float)
            inp_count.append(torch.sum(inp))
            inputs = torch.matmul(inp.unsqueeze(1), anchors).squeeze(1)
            inputs /= (1.0 + torch.sum(inp, dim=-1, keepdim=True))

            layer_id += 1
            anchors = torch.cat((anchors, output), dim=1)
            anchors_w_1 = torch.cat((anchors_w_1, output), dim=1)

            # check stop search condition
            if np.all(sample_count):
                stop = True
            sample_count = np.concatenate((sample_count, np.zeros(shape=[sample_count.shape[0], 1])), axis=1)

        self.sample_arc = arc_seq
        self.seq_string = self._arc_seq_to_string(self.sample_arc)

        entropys = torch.cat(entropys)
        self.sample_entropy = torch.sum(entropys)

        log_probs = torch.cat(log_probs)
        self.sample_log_prob = torch.sum(log_probs)