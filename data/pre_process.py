from tqdm.notebook import tqdm

class PreprocessDataset(object):

    def __init__(self,
                 all_df=None,
                 train_rate=0.8,
                 uid_name='user_id',
                 iid_name='item_id',
                 rating_name='rating'):
        """
        Constructor

        dataset: The dataset to be used, ml-100k, ml-1m, ml-10m, ml-10m
        """
        self.uid_name = uid_name
        self.iid_name = iid_name
        self.rating_name = rating_name

        all_df = PreprocessDataset.drop_invalid_data(all_df, train_rate)
        self.all_df = PreprocessDataset.generate_internal_ids(all_df)

    @staticmethod
    def generate_internal_ids(all_set):
        """
        Map new internal ID for all users and items
        """
        u_ids = all_set['user_id'].unique().tolist()
        i_ids = all_set['item_id'].unique().tolist()

        user_dict = dict(zip(u_ids, [i for i in range(len(u_ids))]))
        item_dict = dict(zip(i_ids, [i for i in range(len(i_ids))]))

        all_set['user_id'] = all_set['user_id'].map(user_dict)
        all_set['item_id'] = all_set['item_id'].map(item_dict)

        return all_set

    @staticmethod
    def drop_invalid_data(all_set, train_rate=0.8):
        """
        Drop invalid data to make sure training set and test set have the same number of users and items
        """
        test_rate = 1 - train_rate
        t_u_num = 0
        t_i_num = 0
        while True:
            if ('u_num' in locals().keys() and u_num == t_u_num) and ('i_num' in locals().keys() and i_num == t_i_num):
                break
            u_num = all_set.groupby('user_id').count().shape[0]
            i_num = all_set.groupby('item_id').count().shape[0]
            all_set = all_set.groupby('user_id').filter(lambda g: len(g) * test_rate >= 1)
            all_set = all_set.groupby('item_id').filter(lambda g: len(g) * test_rate >= 1)
            t_u_num = all_set.groupby('user_id').count().shape[0]
            t_i_num = all_set.groupby('item_id').count().shape[0]

        # if self.most_active_users > 0:
        #     index = all_set.groupby(self.uid_name)[self.rating_name].count().sort_values(ascending=False).head(self.most_active_users).index.values
        #     all_set = all_set[all_set[self.uid_name].isin(index)]

        return all_set.reset_index(drop=True)

    def train_test_split(self, train_rate=0.8, all_dataset=None):
        """
        Split ratings into Training set and Test set

        """
        self.all_set = self.all_df.copy()
        if all_dataset is not None:
            self.all_set = all_dataset.copy()
        grps = self.all_set.groupby(self.uid_name).groups
        test_df_index = list()
        train_df_index = list()

        test_iid = list()
        train_iid = list()

        for key in tqdm(grps):
            count = 0
            local_index = list()
            grp = np.array(list(grps[key]))
            np.random.shuffle(grp)
            n_test = int(len(grp) * (1 - train_rate))
            for i, index in enumerate(grp):
                if count >= n_test:
                    break
                if self.all_set.iloc[index][self.iid_name] in test_iid:
                    if self.all_set.iloc[index][self.iid_name] not in train_iid:
                        train_iid.append(self.all_set.iloc[index][self.iid_name])
                        train_df_index.append(index)
                        local_index.append(i)
                    continue

                test_iid.append(self.all_set.iloc[index][self.iid_name])
                test_df_index.append(index)
                local_index.append(i)
                count += 1

            grp = np.delete(grp, local_index)

            if count < n_test:
                local_index = list()
                for i, index in enumerate(grp):
                    if count >= n_test:
                        break
                    test_iid.append(self.all_set.iloc[index][self.iid_name])
                    test_df_index.append(index)
                    local_index.append(i)
                    count += 1

                grp = np.delete(grp, local_index)

            train_df_index.append(grp)

        test_df_index = np.hstack(np.array(test_df_index, dtype="object"))
        train_df_index = np.hstack(np.array(train_df_index, dtype="object"))

        np.random.shuffle(test_df_index)
        np.random.shuffle(train_df_index)

        train_df, test_df = self.all_set.iloc[train_df_index], self.all_set.iloc[test_df_index]

        iid_idx = np.intersect1d(train_df[self.iid_name].unique(), test_df[self.iid_name].unique())
        uid_idx = np.intersect1d(train_df[self.uid_name].unique(), test_df[self.uid_name].unique())

        train_df = train_df[(train_df[self.uid_name].isin(uid_idx)) & (train_df[self.iid_name].isin(iid_idx))]
        test_df = test_df[(test_df[self.uid_name].isin(uid_idx)) & (test_df[self.iid_name].isin(iid_idx))]

        u_ids = train_df[self.uid_name].unique().tolist()
        i_ids = train_df[self.iid_name].unique().tolist()

        user_dict = dict(zip(u_ids, [i for i in range(len(u_ids))]))
        item_dict = dict(zip(i_ids, [i for i in range(len(i_ids))]))

        train_df[self.uid_name] = train_df[self.uid_name].map(user_dict)
        train_df[self.iid_name] = train_df[self.iid_name].map(item_dict)

        test_df[self.uid_name] = test_df[self.uid_name].map(user_dict)
        test_df[self.iid_name] = test_df[self.iid_name].map(item_dict)


        return train_df, test_df