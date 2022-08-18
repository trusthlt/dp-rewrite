# DP-Rewrite: Towards Reproducibility and Transparency in Differentially Private Text Rewriting

## Description

DP-Rewrite consists of command-line tools (later APIs) to easily perform differentially private text rewriting on a given dataset. Implemented with [PyTorch](https://pytorch.org/), our library provides models and out-of-the-box datasets for running experiments with [differential privacy](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/dwork.pdf) to provide rigorous privacy guarantees. Concretely, we provide tools for pre-training a text rewriting model (most commonly an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder)), rewriting a dataset with differential privacy using this pre-trained model, and running downstream experiments on original and rewritten datasets. Additionally, our library allows for the seamless incorporation of custom datasets and custom models into the framework.

## Installation

```bash
$ sudo apt-get install python3-dev
```

```bash
$ pip install -r requirements.txt
```

## Quick Start: Running Experiments

Steps to running an experiment:
1. Choose a **mode** (`pretrain`, `rewrite`, or `downstream`, see **Modes** below for more details).
2. Choose a **model**. Currently only `adept` is available for pre-training and rewriting, and `bert_downstream` for downstream classification.
3. Choose a desired dataset (see **Custom Datasets** below if you want to use your own dataset).
4. Specify a **name** for your experiment with `--name`. This will be the folder name associated with your experiment where checkpoints, configuration info, learning curves, various statistics and rewritten data will be saved to.
5. Specify two main directories: (1) `--output_dir` where the above folder will be saved and (2) `asset_dir` where datasets will be loaded from. See **Directories** below for more information.
6. Finally, you can configure additional hyperparameters for the experiment such as `--batch_size` and `--learning_rate`. For a complete list of available arguments, you can run `python main.py -h`.

For sample experiment runs, please look at the shell scripts provided in `sample_scripts`. These contain test commands covering the most important basic arguments.

Additionally, if you are on a machine with low resources, you can specify `--local True`, setting `--local_iter_size` to the desired number of iterations per epoch to run the model.

Please read the short introduction about the three available modes below.

### Modes

There are 3 modes of experiments:

1. `pretrain`: This is used for training/fine-tuning an autoencoder model on some data.
2. `rewrite`: This is used for loading a checkpoint of an autoencoder model (provide the checkpoint with param `last_checkpoint_path`) and running it over a whole data set (= rewriting the dataset).
3. `downstream`: This is used for training and/or evaluating a downstream model on some data. Can be done on the raw datasets we provide (see below) or on rewritten data (use parameters `--dataset custom` and provide the associated path with `--custom_train_path`, as well as optionally `--custom_valid_path` and `--custom_test_path`). More information in **Custom Datasets** below.

### Pre-defined Experiments

Optionally, it is possible to specify a pre-defined experiment with `--experiment` which simplifies the above **Quick Start** process by pre-selecting parameters associated with a given experiment (e.g. *mode*, *model*, etc.).

If using a pre-defined experiment, any conflicting parameters specified will be overridden.

The specific list of parameters can be seen in `prepare_specific_experiment` of `utils.py`. The currently available experiments are outlined below:

| Experiment name | Description |
| --- | --- |
| `adept_l2norm_pretrain` | Our implementation of the original pre-training procedure used in [ADePT](https://arxiv.org/abs/2102.01502) |
| `adept_l1norm_pretrain` | Modified ADePT pre-training procedure based on https://arxiv.org/abs/2109.03175 |
| `adept_l2norm_rewrite` | Our implementation of the original rewriting procedure used in [ADePT](https://arxiv.org/abs/2102.01502) |
| `adept_l1norm_rewrite` | Modified ADePT rewriting procedure from https://arxiv.org/abs/2109.03175 |

## Available Datasets

All available datasets will be downloaded and preprocessed automatically to fit the required CVS format.

| Dataset Name | Description | Reference |
|---|---|---|
| `imdb` | Dataset commonly used for binary sentiment classification. Split: 25,000 train, 25,000 test examples. | Original source: https://ai.stanford.edu/~amaas/data/sentiment/ <br>Fetched from: https://huggingface.co/datasets/imdb<br>Paper citation: [Maas et al., 2011](https://aclanthology.org/P11-1015.pdf) |
| `atis` | Airline Travel Information Systems (ATIS) dataset, originally consisting of audio recordings and manual transcripts related to flight reservations, meant for the tasks of intent detection and slot filling. Here we include the manual transcripts and intent labels of the dataset. Split: 4,478 train, 500 development and 893 test examples. | Original source: https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data<br>Paper citation: [Dahl et al., 1994](https://aclanthology.org/H94-1010.pdf) |
| `snips_2016` | Dataset collected for the Snips personal assistant, used for an NLU benchmark from 2016. No pre-defined dataset split, 328 examples in total. | Original source: https://github.com/sonos/nlu-benchmark/tree/master/2016-12-built-in-intents<br>Fetched from: https://huggingface.co/datasets/snips_built_in_intents |
| `snips_2017` | Dataset collected from the Snips personal voice assistant, used for an NLU benchmark from 2017. Split: 13,084 train, 700 development and 700 test examples. | Original source: https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines<br>Paper citation: [Coucke et al., 2018](https://arxiv.org/abs/1805.10190) |
| `wikipedia` | Currently using the Wikipedia dump from May 1, 2020. | https://huggingface.co/datasets/wikipedia |

### Custom Datasets

If you want to use your **own dataset**:
- Please specify `--dataset custom`, along with  `--custom_train_path` (also optionally `--custom_valid_path` and `--custom_test_path`).
- If `--custom_valid_path` is not provided, training and validation sets will be prepared from the training path, with split ratio according to `--train_split_ratio`.
- Datasets should be in CSV format, with one column for 'text', and optionally a second column for 'labels'. If `--prepend_labels` is True, labels will be prepended to each row's text for `pretrain` and `rewrite` modes.
- Test data is not used for the `pretrain` mode.

## Available Models

### Autoencoders

| Model Name | Description | Author |
|---|---|---|
| `adept` | Our reference implementation of Adept. As the Adept paper lacks info about the exact network structure, some architecture decisions were made by us. | Timour |

### Downstream Models

| Model Name | Description | Author |
|---|---|---|
| `bert_downstream` | A BERT model with a feedforward head (output dimension based on the number of classes in the dataset). The specific type of transformer can be specified with `--transformer_type`. | Timour |

### Custom Models

It is also possible to include a custom model for the experiments. The procedure is as follows:

...

## Project Structure

- ``assets`` sub-folders contain the (raw) data of training datasets and embeddings.
- ``models``
    - ``autoencoders`` contains all (PyTorch) autoencoder models that are used to privatize the data, each in its own python file.
    - ``downstream_models`` contains all (PyTorch) models that tackle a downstream task, each in its own python file. They can be used with original or rewritten data.
- ``sample_scripts`` contains tests for different datasets and experiments.
- ``download.py`` can be used to download datasets and embeddings to the assets folder **(currently not used)**.
- ``dataload.py`` is the main dataloading script which prepares existing and custom datasets.
- ``experiments.py`` contains three types of "experiments" and the main training/evaluation loops for the framework.
- ``main.py`` can be executed from the command line with arguments to run experiments (see Quick Start above).
- ``preprocessing.py`` contains preprocessing pipelines that transform the data according to a model's needs. Separate pipelines are implemented for transformer and RNN-based models.
- ``settings.py`` specifies all the arguments that `main.py` can be run with.
- ``utils.py`` contains additional helper functions and classes (e.g. `EarlyStopping`)

## Directories

### Overview

If you want to use the **default folders** in the project structure for data, logs, stats, and so on, you don't need to
specify any paths for using autoencoders and you only need to specify `embed_dir_unprocessed` if you want to train/use
an RNN-based model with pre-trained embeddings (`--embed_type` as `glove` or `word2vec`).

If you want to **load the project in an IDE**, it's a good idea to set `output_dir` & `asset_dir` to paths outside the project folders, as some IDEs will take very long to index large amounts of data that may be written into these folders.

### Details

**Output directories**:

- `output_dir`: Stats & logs (defaults to `<project_root>/results`)
- `dump_dir`: Directory for things that might need much storage (e.g., model checkpoints & rewritten data) (defaults to `output_dir`)
- In each of these directories, data will be written to a custom experiment name (if none is given, this is set to the current timestamp)

**Input directories**:

- `asset_dir`: Where to look for assets like datasets and processed embeddings (defaults to `<project_root>/assets`)
- `embed_dir_unprocessed`: Where to look for pre-trained word embeddings when using RNN-based models.
  - Specify the actual path of the embeddings, not the parent folder.
  - 'Unprocessed' refers to the original word embedding models, not modified for specific experiments (e.g. subsampled vocabularies based on frequency in a given dataset). The 'processed' embedding directory is automatically created under `asset_dir` during preprocessing.
- `data_dir`: If you store your data somewhere else than the other assets, you can specify the path here (defaults to `asset_dir`)

All paths should be specified as global paths.
