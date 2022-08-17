import requests
import shutil
import os, pathlib
from tqdm.auto import tqdm
from datasets import load_dataset


project_assets_folder = os.path.join( pathlib.Path(__file__).parent.resolve(), 'assets' )


def download_asset(name:str, asset_dir:str=project_assets_folder, overwrite:bool=False):
    """ Download asset (such as a dataset or embeddings).

    :param name: Name of the asset.
    :param asset_dir: Parent folder of all assets.
    :param overwrite: Whether the asset should be overwritten if its folder already exists.
    :return:
    """
    data_subdir = os.path.join(asset_dir, name)
    if os.path.exists(data_subdir) and not overwrite:
        raise Exception(f'Folder {data_subdir} exists. Aborting. '
                        f'If you want to overwrite the folder, '
                        f'pass overwrite=True to download_asset.')
    else:
        os.makedirs(data_subdir)
    print(f'The assets will be saved in {data_subdir}')

    if name in ['openwebtext', 'imdb']:
        # native huggingface datasets
        _ = load_dataset(name, cache_dir=data_subdir)
    elif name == 'word2vec_googlenews':
        url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
        save_path = os.path.join(data_subdir, 'GoogleNews-vectors-negative300.bin.gz')
        _download_file(url, save_path)
    else:
        raise NotImplementedError('This asset is not available for download as of now.')


def _download_file(url, save_path):
    print(f'Downloading:')
    print(f'\tfrom: {url}')
    print(f'\tto: {save_path}')
    with requests.get(url, stream=True) as r:
        file_size = int(r.headers.get('Content-Length', 0))
        desc = "(Unknown total file size)" if file_size == 0 else ""
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
            with open(save_path, 'wb+') as f:
                shutil.copyfileobj(r_raw, f)
    print(f'Finished downloading.')
    # return local_filename
    