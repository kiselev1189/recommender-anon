import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import OneHotEncoder
from read_utils import user_item_matrix
from fastFM import als


class SVDRecommender:
    rec_type = 'svd_rec'

    def __init__(self, train_df, n_clusters=6):
        self.X, self.index_user_id, self.index_set_id = user_item_matrix(train_df)
        self.n_clusters = n_clusters

        self.U, self.S, self.V = randomized_svd(self.X, n_components=n_clusters, random_state=0)

        self.cluster_users, self.cluster_recs = self.clusterize()

    def clusterize(self):

        cluster_users = {i: [] for i in range(0, self.n_clusters)}
        for i, user in enumerate(self.U):
            cluster_id = np.argmax(user)
            cluster_users[cluster_id].append(self.index_user_id[i])

        cluster_recs = {i: [] for i in range(0, self.n_clusters)}
        for cluster_id in range(0, self.n_clusters):
            cluster_row = self.V[cluster_id]
            top_sets = [self.index_set_id[i] for i in np.argsort(cluster_row)[::-1]]
            cluster_recs[cluster_id] = top_sets

        return cluster_users, cluster_recs

    def predict(self, user_id):
        user_cluster = None
        for c, cluster in self.cluster_users.iteritems():
            if user_id in cluster:
                user_cluster = c
                break
        return self.cluster_recs[user_cluster]


class FMRecommender:
    rec_type = 'fm_rec'

    def __init__(self, train_df):
        self.one_hot_enc = OneHotEncoder(handle_unknown='ignore')
        self._trained = False
        self.recommender = als.FMClassification(n_iter=400, init_stdev=0.001, rank=30, l2_reg_w=2, l2_reg_V=2)
        # init recommenders
        self.train_df = train_df[['user_id', 'item_set_id']]
        self.train(self.train_df)

    def train(self, df_train_pos, n_epochs=10, lr_start=0.001,
              neg_sample_alpha=6, lr_end=None, warm_start=False):

        self.train_data = df_train_pos


        # fit one hot encoder
        if not warm_start or not self._trained:
            self.one_hot_enc.fit(df_train_pos)

        df_train_neg = self._negative_sampling(df_train_pos, neg_sample_alpha)
        df_train = df_train_pos.append(df_train_neg)[['user_id', 'item_set_id']]
        Y = self._get_Y_train(df_train_pos, df_train_neg, self.recommender.task)
        X = self.one_hot_enc.transform(df_train)

        self.recommender.fit(X, Y)

        self._trained = True

    def predict_(self, X):

        y_proba = self.predict_proba(X)
        y_pred = np.ones_like(y_proba, dtype=np.float64)
        y_pred[y_proba <= .5] = -1
        return y_pred

    def predict_proba(self, X):

        X_one_hot = self.one_hot_enc.transform(X)
        predictions = []
        predictions.append(self.recommender.predict_proba(X_one_hot))

        return np.mean(predictions, axis=0)

    def predict(self, user_id):

        all_places = set(self.train_data['item_set_id'].unique())

        places_to_score = all_places

        user_places = np.empty((len(places_to_score), 2), dtype=int)
        for i, place_id in enumerate(places_to_score):
            user_places[i] = (user_id, place_id)

        scores = self.predict_proba(user_places)
        top_indices = scores.argsort()[::-1]

        return user_places[top_indices, 1]

    def _negative_sampling(self, data, alpha):

        all_places = set(data['item_set_id'].unique())

        negative_samples = []
        for user_id, gr_user in data.groupby('user_id'):
            user_places = set(gr_user['item_set_id'])
            places_to_sample = list(all_places - user_places)
            n_samples = min(int(round(alpha * len(user_places))), len(places_to_sample))

            if n_samples > 0:
                sample_places = np.random.choice(places_to_sample, n_samples, replace=False)
            elif alpha < 0:
                sample_places = places_to_sample
            else:
                continue

            for place_id in sample_places:
                row = {'item_set_id': place_id, 'user_id': user_id}
                negative_samples.append(row)

        return pd.DataFrame(negative_samples)

    def _get_Y_train(self, df_train_pos, df_train_neg, task):

        pos_len = df_train_pos.shape[0]
        neg_len = df_train_neg.shape[0]

        # for ranking do pairs
        if task == 'ranking':
            Y_train = []
            for user_id in df_train_pos.fk_user_id.unique():
                pos_iloc = np.where(df_train_pos.fk_user_id == user_id)[0]
                neg_iloc = np.where(df_train_neg.fk_user_id == user_id)[0]
                for pos_i in pos_iloc:
                    for neg_i in neg_iloc:
                        Y_train.append((pos_i, neg_i + pos_len))
            Y_train = np.array(Y_train)
        # for regression and classification do {-1, 1} encoding
        else:
            Y_train = np.ones(pos_len + neg_len)
            Y_train[pos_len:] = -1

        return Y_train


class MostPopularRecommender:
    rec_type = 'most_popular_rec'

    """Naive recommender for most popular goods.

        Recommends just top sales.
        """

    def __init__(self, train_df):
        self.X, self.index_user_id, self.index_set_id = user_item_matrix(train_df)
        self.most_popular = self.get_most_popular(self.X)

    def get_most_popular(self, X):
        popularity_vector = np.sum(X, axis=0)
        most_pop_sets = [self.index_set_id[i] for i in np.argsort(popularity_vector)[::-1]]
        return most_pop_sets

    def predict(self, user):
        return self.most_popular