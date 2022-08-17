import os
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from copy import deepcopy
from tqdm import tqdm
import pdb
from preprocessing import Preprocessor_for_RNN, Preprocessor_for_Transformer, Custom_Preprocessor

#os.environ["HF_DATASETS_OFFLINE"] = '1'

# 'dataset' classes need to take the following hyperparameters:

# Required:
# dataset_name, in_dir, out_data_dir, data_dir,
# max_seq_len, batch_size, private, local, mod_type,
# mode (might not need, just use subclasses)

# Optional:
# vec_dir, vocab_size, embed_size (RNN-based),
# transformer_type (transformer-based), epsilon (private),
# privatized_validation (downstream mode)

class General_Dataset(ABC):
    def __init__(self, dataset_name, data_dir, checkpoint_dir, max_seq_len,
                 batch_size, train_ratio=0.9, embed_type='glove',
                 embed_size=300, embed_dir_processed=None,
                 embed_dir_unprocessed=None, vocab_size=None,
                 model_type='transformer', private=False,
                 prepend_labels=False, transformer_type='bert-base-uncased',
                 local=False, last_checkpoint_path=False):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
            # main directory where data is stored (all modes; e.g. imdb, yelp)
        self.checkpoint_dir = checkpoint_dir
        self.last_checkpoint_path = last_checkpoint_path
        self.embed_dir_processed = embed_dir_processed
            # vocabulary and embeddings directory (renamed from 'in_dir')
            # used after processing vectors from below 'vec_model_dir'
        self.embed_dir_unprocessed = embed_dir_unprocessed
            # downloaded pre-trained model directory

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.model_type = model_type
        self.embed_type = embed_type
        self.transformer_type = transformer_type
        self.prepend_labels = prepend_labels
        self.train_ratio = train_ratio

        if model_type == 'transformer':
            self.preprocessor = Preprocessor_for_Transformer(
                    checkpoint_dir=checkpoint_dir,
                    transformer_type=transformer_type,
                    max_seq_len=max_seq_len, batch_size=batch_size,
                    prepend_labels=prepend_labels)
        elif model_type == 'rnn':
            if embed_dir_processed is None:
                raise Exception("Please specify 'embed_dir_processed' for RNN-based models.")
            self.preprocessor = Preprocessor_for_RNN(
                    embed_dir_processed, embed_dir_unprocessed,
                    vocab_size=vocab_size, embed_type=embed_type,
                    embed_size=embed_size, checkpoint_dir=checkpoint_dir,
                    max_seq_len=max_seq_len, batch_size=batch_size,
                    prepend_labels=prepend_labels)
        else:
            print("'model_type' not specified, using custom preprocessor...")
            self.preprocessor = Custom_Preprocessor()

        self.private = private
        self.local = local

    @abstractmethod
    def load_and_process(self):
        pass


class Pretrain_Dataset(General_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_size = None
        self.train_iterator = None
        self.valid_iterator = None

    def load_and_process(self, custom_train_path=None,
                         custom_valid_path=None, custom_test_path=None):
        if self.dataset_name == 'imdb':
            print("Preparing IMDb dataset...")
            cache_dir = os.path.join(self.data_dir, 'imdb')
            if not os.path.exists:
                os.makedirs(cache_dir)
            data = load_dataset('imdb', cache_dir=cache_dir)
            data_split = data['train'].train_test_split(
                test_size=(1-self.train_ratio))
            train_data = data_split['train']
            valid_data = data_split['test']
        elif self.dataset_name == 'atis':
            print("Preparing ATIS dataset...")
            train_csv_path = os.path.join(self.data_dir, 'atis', 'atis_train.csv')
            valid_csv_path = os.path.join(self.data_dir, 'atis', 'atis_valid.csv')
            train_data = load_dataset("csv", data_files=train_csv_path,
                                      column_names=["label", "text"])
            valid_data = load_dataset("csv", data_files=valid_csv_path,
                                      column_names=["label", "text"])
            train_data = train_data['train']
            valid_data = valid_data['train']
        elif self.dataset_name == 'snips_2016':
            print("Preparing SNIPS dataset (2016 version)...")
            cache_dir = os.path.join(self.data_dir, 'snips_2016')
            if not os.path.exists:
                os.makedirs(cache_dir)
            data = load_dataset('snips_built_in_intents', cache_dir=cache_dir)
            data_split = data['train'].train_test_split(
                test_size=(1-self.train_ratio))
            train_data = data_split['train']
            valid_data = data_split['test']
        elif self.dataset_name == 'snips_2017':
            print("Preparing SNIPS dataset (2017 version)...")
            train_csv_path = os.path.join(self.data_dir, 'snips_2017',
                                          'snips_train.csv')
            valid_csv_path = os.path.join(self.data_dir, 'snips_2017',
                                          'snips_valid.csv')
            train_data = load_dataset("csv", data_files=train_csv_path,
                                      column_names=["label", "text"])
            valid_data = load_dataset("csv", data_files=valid_csv_path,
                                      column_names=["label", "text"])
            train_data = train_data['train']
            valid_data = valid_data['train']
        elif self.dataset_name == 'wikipedia':
            print("Preparing Wikipedia dataset...")
            cache_dir = os.path.join(self.data_dir, 'wiki')
            if not os.path.exists:
                os.makedirs(cache_dir)
            data = load_dataset("wikipedia", "20200501.en", split='train',
                                cache_dir=cache_dir)
            data = data.add_column("label", np.zeros(len(data)))
            data_split = data.train_test_split(test_size=(1-self.train_ratio))
            train_data = data_split['train']
            valid_data = data_split['test']
        else:
            print("Preparing custom dataset...")
            if custom_train_path is not None:
                train_data = load_dataset('csv', data_files=custom_train_path,
                                          column_names=["label", "text"])
                train_data = train_data['train']
                if np.all(np.array(train_data['text']) == None):
                    # If there is only one column in the CSV file, then the
                    # second column in the dataset will only have None, hence
                    # need to remove it and rename the first column
                    train_data = train_data.remove_columns("text")
                    train_data = train_data.rename_column("label", "text")
                    train_data = train_data.add_column(
                        "label", np.zeros(len(train_data)))
                    if self.prepend_labels:
                        raise Exception("Requested option to prepend labels to each dataset tensor, but provided CSV file has no labels.")
            else:
                raise Exception(f"{self.dataset_name} not in currently prepared datasets, but 'custom_train_path' is None. Please either specify a dataset name among existing datasets, or add a custom dataset path.")
            if custom_valid_path is not None:
                valid_data = load_dataset('csv', data_files=custom_valid_path,
                                          column_names=["label", "text"])
                valid_data = valid_data['train']
                if np.all(np.array(valid_data['text']) == None):
                    # If there is only one column in the CSV file, then the
                    # second column in the dataset will only have None, hence
                    # need to remove it and rename the first column
                    valid_data = valid_data.remove_columns("text")
                    valid_data = valid_data.rename_column("label", "text")
            else:
                # If no validation path specified, make a split from the
                # training set
                train_ratio = self.train_ratio
                full_length = len(train_data)
                train_length = int(full_length * train_ratio)
                val_length = full_length - train_length
                lengths = [train_length, val_length]
                train_data, valid_data = torch.utils.data.random_split(
                    train_data, lengths)
                train_data = train_data.dataset
                valid_data = valid_data.dataset

        if self.embed_type not in ['none'] or self.model_type == 'transformer':
            train_data, valid_data, _ = self.preprocessor.process_data(
                    train_data, valid_data)
            # len(train_data): length of train dataset split
            # train_data[i][0]: torch tensor of max_seq_len
            # train_data[i][1]: length of tensor
            # train_data[i][2]: label string
        else:
            train_data, valid_data, _ =\
                    self.preprocessor.process_data_no_embeds(train_data,
                                                             valid_data)
        if self.local:
            shuffle = False
        else:
            shuffle = True

        train_dataloader = DataLoader(train_data, batch_size=self.batch_size,
                                      shuffle=shuffle)
        valid_dataloader = DataLoader(valid_data, batch_size=self.batch_size,
                                      shuffle=shuffle)

        self.sample_size = len(train_data)
        print('Num training:', self.sample_size)
        print('Num validation:', len(valid_data))

        self.train_iterator = train_dataloader
        self.valid_iterator = valid_dataloader


class Rewrite_Dataset(General_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
#        self.rewritten_data_dir = rewritten_data_dir
            # rewritten data directory (renamed from 'out_data_dir')
        self.sample_size = None
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

    def load_and_process(self, custom_train_path=None,
                         custom_valid_path=None, custom_test_path=None):
        if self.dataset_name == 'imdb':
            print("Preparing IMDb dataset...")
            cache_dir = os.path.join(self.data_dir, 'imdb')
            if not os.path.exists:
                os.makedirs(cache_dir)
            data = load_dataset('imdb', cache_dir=cache_dir)
            train_data = data['train']
            valid_data = None
            test_data = data['test']
        elif self.dataset_name == 'atis':
            print("Preparing ATIS dataset...")
            train_csv_path = os.path.join(self.data_dir, 'atis',
                                          'atis_train.csv')
            valid_csv_path = os.path.join(self.data_dir, 'atis',
                                          'atis_valid.csv')
            test_csv_path = os.path.join(self.data_dir, 'atis',
                                         'atis_test.csv')
            train_data = load_dataset("csv", data_files=train_csv_path,
                                      column_names=["label", "text"])
            valid_data = load_dataset("csv", data_files=valid_csv_path,
                                      column_names=["label", "text"])
            test_data = load_dataset("csv", data_files=test_csv_path,
                                     column_names=["label", "text"])
            train_data = train_data['train']
            valid_data = valid_data['train']
            test_data = test_data['train']
        elif self.dataset_name == 'snips_2016':
            print("Preparing SNIPS dataset (2016 version)...")
            cache_dir = os.path.join(self.data_dir, 'snips_2016')
            if not os.path.exists:
                os.makedirs(cache_dir)
            data = load_dataset('snips_built_in_intents', cache_dir=cache_dir)
            train_data = data['train']
            valid_data = None
            test_data = None
        elif self.dataset_name == 'snips_2017':
            print("Preparing SNIPS dataset (2017 version)...")
            train_csv_path = os.path.join(self.data_dir, 'snips_2017',
                                          'snips_train.csv')
            valid_csv_path = os.path.join(self.data_dir, 'snips_2017',
                                          'snips_valid.csv')
            test_csv_path = os.path.join(self.data_dir, 'snips_2017',
                                         'snips_test.csv')
            train_data = load_dataset("csv", data_files=train_csv_path,
                                      column_names=["label", "text"])
            valid_data = load_dataset("csv", data_files=valid_csv_path,
                                      column_names=["label", "text"])
            test_data = load_dataset("csv", data_files=test_csv_path,
                                     column_names=["label", "text"])
            train_data = train_data['train']
            valid_data = valid_data['train']
            test_data = test_data['train']
        else:
            print("Preparing custom dataset...")
            if custom_train_path is not None:
                train_data = load_dataset('csv', data_files=custom_train_path,
                                          column_names=["label", "text"])
                train_data = train_data['train']
                if np.all(np.array(train_data['text']) == None):
                    # If there is only one column in the CSV file, then the
                    # second column in the dataset will only have None, hence
                    # need to remove it and rename the first column
                    train_data = train_data.remove_columns("text")
                    train_data = train_data.rename_column("label", "text")
                    train_data = train_data.add_column("label", np.zeros(len(train_data)))
                    if self.prepend_labels:
                        raise Exception("Requested option to prepend labels to each dataset tensor, but provided CSV file has no labels.")
            else:
                raise Exception(f"{self.dataset_name} not in currently prepared datasets, but 'custom_train_path' is None. Please either specify a dataset name among existing datasets, or add a custom dataset path.")
            if custom_valid_path is not None:
                valid_data = load_dataset('csv', data_files=custom_valid_path,
                                          column_names=["label", "text"])
                valid_data = valid_data['train']
                if np.all(np.array(valid_data['text']) == None):
                    # If there is only one column in the CSV file, then the
                    # second column in the dataset will only have None, hence
                    # need to remove it and rename the first column
                    valid_data = valid_data.remove_columns("text")
                    valid_data = valid_data.rename_column("label", "text")
            else:
                valid_data = None
            if custom_test_path is not None:
                test_data = load_dataset('csv', data_files=custom_test_path,
                                         column_names=["label", "text"])
                test_data = test_data['train']
                if np.all(np.array(test_data['text']) == None):
                    # If there is only one column in the CSV file, then the
                    # second column in the dataset will only have None, hence
                    # need to remove it and rename the first column
                    valid_data = test_data.remove_columns("text")
                    valid_data = test_data.rename_column("label", "text")
            else:
                test_data = None

        if self.embed_type not in ['none'] or self.model_type == 'transformer':
            train_data, valid_data, test_data = self.preprocessor.process_data(
                    train_data, valid_data=valid_data, test_data=test_data,
                    rewriting=True,
                    last_checkpoint_path=self.last_checkpoint_path)
            # len(train_data): length of train dataset split
            # train_data[i][0]: torch tensor of max_seq_len
            # train_data[i][1]: length of tensor
            # train_data[i][2]: label string
        else:
            train_data, valid_data, test_data =\
                    self.preprocessor.process_data_no_embeds(
                            train_data, valid_data=valid_data,
                            test_data=test_data, rewriting=True,
                            last_checkpoint_path=self.last_checkpoint_path)

        shuffle = False
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size,
                                      shuffle=shuffle)
        self.train_iterator = train_dataloader

        self.sample_size = len(train_data)
        print('Num training:', self.sample_size)

        if valid_data is not None:
            valid_dataloader = DataLoader(valid_data,
                                          batch_size=self.batch_size,
                                          shuffle=shuffle)
            self.valid_iterator = valid_dataloader
            print('Num validation:', len(valid_data))
        if test_data is not None:
            test_dataloader = DataLoader(test_data, batch_size=self.batch_size,
                                         shuffle=shuffle)
            self.test_iterator = test_dataloader
            print('Num test:', len(test_data))


class Downstream_Dataset(General_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_size = None
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

    def load_and_process(self, custom_train_path=None,
                         custom_valid_path=None, custom_test_path=None):
        if self.dataset_name == 'imdb':
            print("Preparing IMDb dataset...")
            cache_dir = os.path.join(self.data_dir, 'imdb')
            if not os.path.exists:
                os.makedirs(cache_dir)
            data = load_dataset('imdb', cache_dir=cache_dir)
            data_split = data['train'].train_test_split(
                test_size=(1-self.train_ratio))
            train_data = data_split['train']
            valid_data = data_split['test']
            test_data = data['test']
        elif self.dataset_name == 'atis':
            print("Preparing ATIS dataset...")
            train_csv_path = os.path.join(self.data_dir, 'atis',
                                          'atis_train.csv')
            valid_csv_path = os.path.join(self.data_dir, 'atis',
                                          'atis_valid.csv')
            test_csv_path = os.path.join(self.data_dir, 'atis',
                                         'atis_test.csv')
            train_data = load_dataset("csv", data_files=train_csv_path,
                                      column_names=["label", "text"])
            valid_data = load_dataset("csv", data_files=valid_csv_path,
                                      column_names=["label", "text"])
            test_data = load_dataset("csv", data_files=test_csv_path,
                                     column_names=["label", "text"])
            train_data = train_data['train']
            valid_data = valid_data['train']
            test_data = test_data['train']
        elif self.dataset_name == 'snips_2016':
            print("Preparing SNIPS dataset (2016 version)...")
            cache_dir = os.path.join(self.data_dir, 'snips_2016')
            if not os.path.exists:
                os.makedirs(cache_dir)
            data = load_dataset('snips_built_in_intents', cache_dir=cache_dir)
            data_split = data['train'].train_test_split(
                test_size=(1-self.train_ratio))
            train_data = data_split['train']
            valid_data = data_split['test']
            test_data = None
        elif self.dataset_name == 'snips_2017':
            print("Preparing SNIPS dataset (2017 version)...")
            train_csv_path = os.path.join(self.data_dir, 'snips_2017',
                                          'snips_train.csv')
            valid_csv_path = os.path.join(self.data_dir, 'snips_2017',
                                          'snips_valid.csv')
            test_csv_path = os.path.join(self.data_dir, 'snips_2017',
                                         'snips_test.csv')
            train_data = load_dataset("csv", data_files=train_csv_path,
                                      column_names=["label", "text"])
            valid_data = load_dataset("csv", data_files=valid_csv_path,
                                      column_names=["label", "text"])
            test_data = load_dataset("csv", data_files=test_csv_path,
                                     column_names=["label", "text"])
            train_data = train_data['train']
            valid_data = valid_data['train']
            test_data = test_data['train']
        else:
            print("Preparing custom dataset...")
            if custom_train_path is not None:
                train_data = load_dataset('csv', data_files=custom_train_path,
                                          column_names=["label", "text"])
                train_data = train_data['train']
                if np.all(np.array(train_data['text']) == None):
                    # If there is only one column in the CSV file, then the
                    # second column in the dataset will only have None, hence
                    # need to remove it and rename the first column
                    train_data = train_data.remove_columns("text")
                    train_data = train_data.rename_column("label", "text")
                    if self.prepend_labels:
                        raise Exception("Requested option to prepend labels to each dataset tensor, but provided CSV file has no labels.")
            else:
                raise Exception(f"{self.dataset_name} not in currently prepared datasets, but 'custom_train_path' is None. Please either specify a dataset name among existing datasets, or add a custom dataset path.")
            if custom_valid_path is not None and custom_valid_path.lower() != 'none':
                valid_data = load_dataset('csv', data_files=custom_valid_path,
                                          column_names=["label", "text"])
                valid_data = valid_data['train']
                if np.all(np.array(valid_data['text']) == None):
                    # If there is only one column in the CSV file, then the
                    # second column in the dataset will only have None, hence
                    # need to remove it and rename the first column
                    valid_data = valid_data.remove_columns("text")
                    valid_data = valid_data.rename_column("label", "text")
            else:
                data_split = train_data.train_test_split(
                    test_size=(1-self.train_ratio))
                train_data = data_split['train']
                valid_data = data_split['test']
            if custom_test_path is not None and custom_test_path.lower() != 'none':
                test_data = load_dataset('csv', data_files=custom_test_path,
                                         column_names=["label", "text"])
                test_data = test_data['train']
                if np.all(np.array(test_data['text']) == None):
                    # If there is only one column in the CSV file, then the
                    # second column in the dataset will only have None, hence
                    # need to remove it and rename the first column
                    valid_data = test_data.remove_columns("text")
                    valid_data = test_data.rename_column("label", "text")
            else:
                test_data = None

        if self.embed_type not in ['none'] or self.model_type == 'transformer':
            train_data, valid_data, test_data = self.preprocessor.process_data(
                    train_data, valid_data=valid_data, test_data=test_data)
            # len(train_data): length of train dataset split
            # train_data[i][0]: torch tensor of max_seq_len
            # train_data[i][1]: length of tensor
            # train_data[i][2]: label string
        else:
            train_data, valid_data, test_data =\
                    self.preprocessor.process_data_no_embeds(
                            train_data, valid_data=valid_data,
                            test_data=test_data)

        if self.local:
            shuffle = False
        else:
            shuffle = True

        train_dataloader = DataLoader(train_data, batch_size=self.batch_size,
                                      shuffle=shuffle)
        self.train_iterator = train_dataloader

        self.sample_size = len(train_data)
        print('Num training:', self.sample_size)

        valid_dataloader = DataLoader(valid_data,
                                      batch_size=self.batch_size,
                                      shuffle=shuffle)
        self.valid_iterator = valid_dataloader
        print('Num validation:', len(valid_data))

        if test_data is not None:
            test_dataloader = DataLoader(test_data, batch_size=self.batch_size,
                                         shuffle=shuffle)
            self.test_iterator = test_dataloader
            print('Num test:', len(test_data))
