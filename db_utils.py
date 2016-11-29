# -*- coding: utf-8 -*-
import pymysql
import pandas as pd
from read_utils import identify_successors

connection = pymysql.connect(user="PLACEHOLDER", password="PLACEHOLDER", host="PLACEHOLDER", database="invisible2")

"""This module defines utilities for database interaction"""

def train_data():
    """Downloads all transactions, removes free goods, gift certificates and transactions made by staff"""
    query = open("PLACEHOLDER").read()
    staff_users = [int(i) for i in list(open("staff_and_test_user_id").readlines())[1:]]
    df = pd.read_sql(query, connection)

    df['order_date'] = pd.to_datetime(df['order_date'])
    df['order_fulfilment_date'] = pd.to_datetime(df['order_fulfilment_date'])
    df['user_invite_requested'] = pd.to_datetime(df['user_invite_requested'])
    df['user_activated'] = pd.to_datetime(df['user_activated'])

    df = df[df["set_price"] > 0.0]  # removes free glasses
    df = identify_successors(df)

    df = df[~df["set_title"].str.contains(u"Сертификат")]
    df = df[~df["user_id"].isin(staff_users)]
    return df


def current_sets():
    """Downloads and returns a set of goods that are availiable"""
    query = open("current_sets.sql").read()
    df = pd.read_sql(query, connection)
    return set(df["item_set_id"].unique())


def featured_sets(current):
    """Returns a whitelist of goods that can be shown on landing page"""
    whitelist = pd.read_csv("whitelist.csv")[["ID", "Новый сет ID", "Тематические"]]
    whitelist["Новый сет ID"].fillna(whitelist["ID"], inplace=True)
    del whitelist["ID"]
    whitelist = whitelist.dropna()
    whitelist = set([int(i) for i in whitelist["Новый сет ID"].unique()])
    return current.intersection(whitelist)
