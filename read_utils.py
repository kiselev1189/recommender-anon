# -*- coding: utf-8 -*-
import pandas as pd
import math
import numpy as np
from collections import Counter
import datetime as dt


def filter_new(train, test):
    """Removes previously unseen users from test set.

    Parameters
    ----------
    train : pandas.dataframe
    test : pandas.dataframe

    Returns
    -------
    test : pandas.dataframe
        Filtered dataframe for users which are new.

    """

    train_ids = train["user_id"].unique()
    test = test[test["user_id"].isin(train_ids)]
    print "Users:", len(train["user_id"].unique())

    train_ids = train["item_set_id"].unique()
    test = test[test["item_set_id"].isin(train_ids)]
    print "Sets:", len(train["item_set_id"].unique())

    return test


def clean_sold_out(df, test_start, offset):
    """Cleans out sets, which were moved out of sale.

    Parameters
    ----------
    df : pandas.dataframe
        Input dataframe.
    test_start : dt.datetime
        When test set starts.
    offset : dt.datetime
        Offset to make sure we cense the set right.

    Returns
    -------
    df : pandas.dataframe
        Cleaned dataframe

    """

    sales_start = test_start - offset
    sales_end = test_start
    current_sets = df[(df["order_date"] >= sales_start) & (df["order_date"] <= sales_end)]
    current_sets = current_sets["item_set_id"].unique()
    print "Active sets:", len(current_sets)
    df = df[df["item_set_id"].isin(current_sets)]
    return df


def identify_successors(df):
    """ Sets all set_ids to set_id of latest successor in each "family"

    Parameters
    ----------
    df : pandas.dataframe
        Dataframe to clean

    Returns
    -------
    df : pandas.dataframe
        Cleaned dataframe
    """

    set_ids = df["item_set_id"].unique()
    set_successors = {}  # set_id: final_successor - int or None

    # shit code to detect possible ancestor-successor circles, implies that successor_id > ancestor_id
    for set_id in set_ids:

        cur_set_id = int(set_id)

        if cur_set_id in set_successors.keys():
            continue
        set_successors[int(cur_set_id)] = None

        chain = []
        while True:

            if len(df[df["item_set_id"] == cur_set_id]) == 0:
                break
            successor = df[df["item_set_id"] == cur_set_id]["set_successor"].iloc[0]

            if math.isnan(successor):
                break
            else:
                chain.append(int(cur_set_id))
                if successor < cur_set_id:
                    break
                cur_set_id = int(successor)

        for i in chain:
            if i != cur_set_id:
                set_successors[i] = cur_set_id

    for set_id in set_successors:
        if set_successors[set_id] is not None:
            df.ix[df["item_set_id"] == set_id, "item_set_id"] = set_successors[set_id]

    return df


def train_test(input_file_path, current_date, test_size):

    """Reads from csv in input_file_path to dataframe, decides which sets are active, cleans data and returns
        train and test dataframes.

    Parameters
    ----------
    input_file_path : str
        Path to file, required
    test_size : pandas.DateOffset
        Size of test set, required
    Returns
    -------
    train, test : tuple
        Tuple of pandas.dataframes

    """

    df = pd.read_csv(input_file_path, encoding="utf-8", low_memory=False)

    df['order_date'] = pd.to_datetime(df['order_date'])
    df['order_fulfilment_date'] = pd.to_datetime(df['order_fulfilment_date'])
    df['user_invite_requested'] = pd.to_datetime(df['user_invite_requested'])
    df['user_activated'] = pd.to_datetime(df['user_activated'])

    df = df[df["order_date"] <= current_date]  # folds

    df = df[df["set_price"] > 0.0]  # removes free glasses
    df = identify_successors(df)

    test_start = df["order_date"].max() - test_size

    print dt.datetime.now(), test_start

    df = df[~df["set_title"].str.contains(u"Сертификат")]
    test = df[df["order_date"] > test_start]
    train = df[df["order_date"] <= test_start]

    test = filter_new(train, test)

    return train, test


def secondary_purchases_top(df):
    """Creates a list of best sellers by secondary purchases

    Parameters
    ----------
    df : pandas.dataframe
    Input dataframe

    Returns
    -------
    list
    """
    purchases = []
    for user, group in df.groupby(["user_id"]):
        group.sort('order_date', ascending=True)
        purchases.extend([i for i in group[1:]["item_set_id"]])
    return [i[0] for i in Counter(purchases).most_common()]


def user_item_matrix(df):
    """Constructs User-Item matrix from dataframe, returns user-item matrix,
                                                           index_user_id dict  - mapping of indexes to users
                                                           index_set_id dict - mapping of indexes to set_ids

    Parameters
    ----------
    df : pandas.dataframe
        Input dataframe

    Returns
    -------
    matrix, index_user_id, index_set_id : tuple

    """

    unique_users = df["user_id"].unique()
    unique_sets = df["item_set_id"].unique()

    matrix = np.zeros((len(unique_users), len(unique_sets)))

    user_id_index = {user_id: i for (user_id, i) in zip(unique_users, range(0, len(unique_users)))}
    set_id_index = {set_id: i for (set_id, i) in zip(unique_sets, range(0, len(unique_sets)))}

    index_user_id = {i: user_id for (user_id, i) in zip(unique_users, range(0, len(unique_users)))}
    index_set_id = {i: set_id for (set_id, i) in zip(unique_sets, range(0, len(unique_sets)))}

    for row in df.iterrows():
        order = row[1]
        user_id = order["user_id"]
        set_id = order["item_set_id"]
        matrix[user_id_index[user_id]][set_id_index[set_id]] = 1  # order["quantity_of_this_set"]

    return matrix, index_user_id, index_set_id


def user_item_matrix_time_weighted(df, time_func, time_func_params):
    """Constructs User-Item matrix from dataframe, returns user-item matrix,
                                                           index_user_id dict  - mapping of indexes to users
                                                           index_set_id dict - mapping of indexes to set_ids

    Parameters
    ----------
    df : pandas.dataframe
        Input dataframe
    time_func : callable
        Function to calculate user_item matrix value from purchase date
    Returns
    -------
    matrix, index_user_id, index_set_id : tuple

    """

    unique_users = df["user_id"].unique()
    unique_sets = df["item_set_id"].unique()

    matrix = np.zeros((len(unique_users), len(unique_sets)))

    user_id_index = {user_id: i for (user_id, i) in zip(unique_users, range(0, len(unique_users)))}
    set_id_index = {set_id: i for (set_id, i) in zip(unique_sets, range(0, len(unique_sets)))}

    index_user_id = {i: user_id for (user_id, i) in zip(unique_users, range(0, len(unique_users)))}
    index_set_id = {i: set_id for (set_id, i) in zip(unique_sets, range(0, len(unique_sets)))}

    for row in df.iterrows():
        order = row[1]
        user_id = order["user_id"]
        set_id = order["item_set_id"]
        date = order["order_date"]
        matrix[user_id_index[user_id]][set_id_index[set_id]] += time_func(date,
                                                                          time_func_params)  # order["quantity_of_this_set"]

    return matrix, index_user_id, index_set_id