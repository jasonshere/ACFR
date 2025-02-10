import tensorflow as tf
import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

DATASETS = {
    'ml-100k': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'ml-1m': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'ml-25m': 'https://files.grouplens.org/datasets/movielens/ml-25m.zip',
    'Ciao': 'https://www.cse.msu.edu/~tangjili/datasetcode/ciao.zip',
    'Ciao2': 'https://www.cse.msu.edu/~tangjili/datasetcode/ciao_with_rating_timestamp.zip',
    'CiaoDVD': 'https://guoguibing.github.io/librec/datasets/CiaoDVD.zip',
    'Amazon-instant-video': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Amazon_Instant_Video.csv',
    'Amazon-music-instruments': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Musical_Instruments.csv',
    'Amazon-baby': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Baby.csv',
    # 'Amazin-apps-and-android': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Apps_for_Android.csv',
    'Amazon-video-games': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Video_Games.csv',
    'Amazon-software': 'http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Software.csv',
    'Amazon-beauty': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv',
    'Amazon-prime-pantry': 'http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Prime_Pantry.csv',
    'Amazon-pet-supplies': 'http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Pet_Supplies.csv',
    'Amazon-toys-and-games': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Toys_and_Games.csv',
    'Amazon-appliances': 'http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Appliances.csv',
    'Amazon-fashion': 'http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION.csv',
    'Amazon-gift-cards': 'http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Gift_Cards.csv',
    'Amazon-musicial-instruments': 'http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Musical_Instruments.csv',
    'Amazon-digital-music': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Digital_Music.csv',
    'Amazon-automative': 'http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Automotive.csv',
    'Amazon-arts': 'http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Arts_Crafts_and_Sewing.csv',
    'Amazon-health': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Health_and_Personal_Care.csv',
    'Amazon-cellphone': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Cell_Phones_and_Accessories.csv',
    'Amazon-sports': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Sports_and_Outdoors.csv',
    'Amazon-cds-and-vinyl': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_CDs_and_Vinyl.csv',
    'Amazon-patio': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Patio_Lawn_and_Garden.csv',
    'Amazon-grocery': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Grocery_and_Gourmet_Food.csv',
    'Amazon-office-products': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Office_Products.csv',
    'Amazon-apps': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Apps_for_Android.csv',
    'Amazon-tools': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Tools_and_Home_Improvement.csv',
}

MD5_HASH = {
    'ml-100k': '0e33842e24a9c977be4e0107933c0723',
    'ml-1m': 'c4d9eecfca2ab87c1945afe126590906',
    'ml-25m': '6b51fb2759a8657d3bfcbfc42b592ada',
}

class AbstractDatasetLoader(ABC):

    def __init__(self, ds_name=None, verbose=False):
        if ds_name is None:
            self.dataset_name = self._dataset_name()
        else:
            self.dataset_name = ds_name
        print("dn: {}" .format(self.dataset_name))
        self.data_dir = self._download_zip_file()
        if verbose:
            self._show_all_files()
        super().__init__()

    def _download_zip_file(self):
        dataset_file = tf.keras.utils.get_file(fname=os.path.basename(DATASETS[self.dataset_name]),
                                       origin=DATASETS[self.dataset_name],
                                       extract=True,
                                       md5_hash=MD5_HASH[self.dataset_name])

        return os.path.splitext(dataset_file)[0]

    def _dataset_name(self):
        if self.__class__.__name__.startswith('ML'):
            return self.__class__.__name__.lower().replace('ml', 'ml-')[0:-13]

        if self.__class__.__name__.startswith('Ciao'):
            return 'Ciao'

    def _show_all_files(self):
        # r=root, d=directories, f = files
        for r, d, f in os.walk(self.data_dir):
            for file in f:
                print(file)

    def _read_table(self, filename, sep, names, encoding='latin-1'):
        return pd.read_csv(os.path.join(self.data_dir, filename),
                    sep=sep,
                    names=names,
                    encoding=encoding)

    def ratings(self):
        return self._read_table(self.RATINGS, self.RATINGS_SEPARATOR, self.RATINGS_COLUMN_NAMES)

    def user_info(self):
        return self._read_table(self.USERS, self.USERS_SEPARATOR, self.USERS_COLUMN_NAMES)

    def item_info(self):
        return self._read_table(self.ITEMS, self.ITEMS_SEPARATOR, self.ITEMS_COLUMN_NAMES)

    def n_ratings(self):
        return self.ratings().shape[0]

    def n_users(self):
        return self.user_info().shape[0]

    def n_items(self):
        return self.item_info().shape[0]

    def sparsity(self):
        return (self.n_users() * self.n_items() - self.n_ratings()) / (self.n_users() * self.n_items())

class ML1MDatasetLoader(AbstractDatasetLoader):

    RATINGS = 'ratings.dat'
    USERS = 'users.dat'
    ITEMS = 'movies.dat'

    RATINGS_SEPARATOR = '::'
    USERS_SEPARATOR = '::'
    ITEMS_SEPARATOR = '::'

    RATINGS_COLUMN_NAMES = ['user_id', 'item_id', 'rating', 'timestamp']
    USERS_COLUMN_NAMES = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    ITEMS_COLUMN_NAMES = ['item_id', 'title', 'release_date', 'video_release_date',
                          'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                          "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                          'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery',
                          'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']

    def __init__(self, ds_name=None, verbose=False):
        super().__init__(ds_name, verbose)

class ML25MDatasetLoader(AbstractDatasetLoader):

    RATINGS = 'ratings.csv'
    USERS = 'users.dat'
    ITEMS = 'movies.dat'

    RATINGS_SEPARATOR = ','
    USERS_SEPARATOR = ','
    ITEMS_SEPARATOR = ','

    RATINGS_COLUMN_NAMES = ['user_id', 'item_id', 'rating', 'timestamp']
    USERS_COLUMN_NAMES = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    ITEMS_COLUMN_NAMES = ['item_id', 'title', 'release_date', 'video_release_date',
                          'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                          "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                          'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery',
                          'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']

    def __init__(self, ds_name=None, verbose=False):
        super().__init__(ds_name, verbose)

    def _read_table(self, filename, sep, names, encoding='latin-1'):
        return pd.read_csv(os.path.join(self.data_dir, filename),
                    sep=sep,
                    skiprows=1,
                    names=names,
                    encoding=encoding)

class ML100KDatasetLoader(AbstractDatasetLoader):

    RATINGS = 'u.data'
    USERS = 'u.user'
    ITEMS = 'u.item'

    RATINGS_SEPARATOR = '\t'
    USERS_SEPARATOR = '|'
    ITEMS_SEPARATOR = '|'

    RATINGS_COLUMN_NAMES = ['user_id', 'item_id', 'rating', 'timestamp']
    USERS_COLUMN_NAMES = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    ITEMS_COLUMN_NAMES = ['item_id', 'title', 'release_date', 'video_release_date',
                          'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                          "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                          'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery',
                          'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']

    def __init__(self, ds_name=None, verbose=False):
        super().__init__(ds_name, verbose)

class AmazonDatasetsLoader():
    def ratings(self, ds_name, UN=20, IN=10):
        dataset_file = tf.keras.utils.get_file(fname=os.path.basename(DATASETS[ds_name]),
                                       origin=DATASETS[ds_name],
                                       extract=False)

        rts = pd.read_csv(dataset_file, sep=',', names=['user_id', 'item_id', 'rating', 'timestamp'])
        # rts = pd.read_csv(dataset_file, sep=',', names=['item_id', 'user_id', 'rating', 'timestamp'])
        rts = rts.drop_duplicates(subset=['item_id', 'user_id'], keep='last')
        while True:
            rts = rts[rts.groupby('user_id')['user_id'].transform('count') >= UN]
            rts = rts[rts.groupby('item_id')['item_id'].transform('count') >= IN]
            print(rts.groupby('user_id').count().min().rating, rts.groupby('item_id').count().min().rating)
            if (rts.groupby('user_id').count().min().rating >= UN) and rts.groupby('item_id').count().min().rating >= IN:
                break
            if np.isnan(rts.groupby('user_id').count().min().rating) or np.isnan(rts.groupby('item_id').count().min().rating):
                break
        # return rts[rts.user_id.isin(uids) & rts.item_id.isin(iids)]
        return rts