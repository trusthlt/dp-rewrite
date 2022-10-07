import torch
from torch.utils.data import Dataset, TensorDataset
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import json
import os
import pdb


def get_model_type(model):
    '''
    Given a specified model for an experiment, return the type of model ('rnn'
    or 'transformer')
    '''
    model_to_model_type = {
        'adept': 'rnn',
        'custom_rnn': 'rnn',
        'custom_transformer': 'transformer',
        'bert_downstream': 'transformer'
        }
    if model in model_to_model_type.keys():
        model_type = model_to_model_type[model]
    else:
        raise Exception("Specified model not in current list of available "
                        "models.")
    return model_type


class EarlyStopping(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, checkpoint_dict, mod_name):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            torch.save(checkpoint_dict, mod_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of '
                  f'{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            checkpoint_dict['checkpoint_early_stopping'] = self.counter
            torch.save(checkpoint_dict, mod_name)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def decode_rewritten(rewritten, preprocessor, remove_special_tokens=True,
                     labels=False, model_type='transformer'):
    if model_type == 'rnn':
        decoded = decode_rewritten_rnn(
            rewritten, preprocessor,
            remove_special_tokens=remove_special_tokens, labels=labels)
    else:
        decoded = decode_rewritten_transformer(
            rewritten, preprocessor,
            remove_special_tokens=remove_special_tokens, labels=labels)

    return decoded


def decode_rewritten_rnn(rewritten, preprocessor, remove_special_tokens=True,
                         labels=False):
    '''
    rewritten: torch tensor size batch X max_seq_len-1, type int64
    preprocessor: preprocessing class from preprocessing.py
    remove_special_tokens: ignore <pad>, <unk>, <sos> and <eos> tokens

    decoded: list of strings, with predicted tokens separated by a space
    '''
    special_tokens = [0, 1, 2, 3]

    decoded = []
    for batch_idx in range(rewritten.shape[0]):
        batch = rewritten[batch_idx, :]
        if remove_special_tokens:
            decoded_batch = [preprocessor.idx2word[idx.item()] for idx in batch if idx not in special_tokens]
        else:
            decoded_batch = [preprocessor.idx2word[idx.item()] for idx in batch]
        decoded.append(decoded_batch)

    if not labels:
        decoded = [' '.join(batch) for batch in decoded]

    # For empty strings
    decoded = [doc if doc != '' else 'UNK' for doc in decoded]

    return decoded


def decode_rewritten_transformer(rewritten, preprocessor,
                                 remove_special_tokens=True, labels=False):
    '''
    rewritten: torch tensor size batch X max_seq_len-1, type int64
    preprocessor: preprocessing class from preprocessing.py
    remove_special_tokens: ignore special tokens according to huggingface's
                           tokenizer

    decoded: list of strings, with predicted tokens separated by a space
    '''
    decoded = preprocessor.tokenizer.batch_decode(
        rewritten, skip_special_tokens=remove_special_tokens)

    if labels:
        raise NotImplementedError

    return decoded


def prepare_specific_experiment(ss, experiment='adept_l1norm_pretrain'):
    '''
    Sets up arguments to fit a given experiment, overwrites required default
    values to properly fit with the experiment.
    For more customizability, can leave 'experiment' as None and select one's
    own parameters for the experiment.
    E.g. 'adept'
    '''
    if experiment == 'adept_l1norm_pretrain':
        ss.args.mode = 'pretrain'
        ss.args.model = 'adept'
        ss.args.model_type = 'rnn'
        ss.args.no_clipping = False
        ss.args.prepend_labels = False
        ss.args.private = False
        ss.args.l_norm = 1
    if experiment == 'adept_l2norm_pretrain':
        ss.args.mode = 'pretrain'
        ss.args.model = 'adept'
        ss.args.model_type = 'rnn'
        ss.args.no_clipping = False
        ss.args.prepend_labels = False
        ss.args.private = False
        ss.args.l_norm = 2
    if experiment == 'adept_l1norm_rewrite':
        ss.args.mode = 'rewrite'
        ss.args.model = 'adept'
        ss.args.model_type = 'rnn'
        ss.args.prepend_labels = False
        ss.args.private = True
        ss.args.l_norm = 1
    if experiment == 'adept_l2norm_rewrite':
        ss.args.mode = 'rewrite'
        ss.args.model = 'adept'
        ss.args.model_type = 'rnn'
        ss.args.prepend_labels = False
        ss.args.private = True
        ss.args.l_norm = 2

    return ss
