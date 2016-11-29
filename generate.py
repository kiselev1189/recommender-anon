from recommenders import SVDRecommender, FMRecommender
from db_utils import current_sets, train_data, featured_sets
from filter import recommend
import pandas as pd
import pymysql
import datetime
import json

"""Generates recommedations and uploads them to database"""


connection = pymysql.connect(user="PLACEHOLDER", password="PLACEHOLDER", host="PLACEHOLDER", database="PLACEHOLDER")


def generate_recommendations(RecommenderClass, train_df, alg_id, current=None, featured=None):
    rec = RecommenderClass(train_df)

    users = train_df["user_id"].unique()
    recs = []
    for user in users:
        recs.append({"user_id": int(user),
                     "recs_offers": json.dumps([int(i) for i in recommend(user, rec, 6, whitelist=current)]),
                     "recs_featured": json.dumps([int(i) for i in recommend(user, rec, 6, whitelist=featured)]),
                     "alg_id": alg_id, "date": datetime.datetime.now(), "subgroup_id": 0})
    return recs


train_df = train_data()
current = current_sets()
featured = featured_sets(current)

recommenders = {"svd_old": SVDRecommender, "fm_v1": FMRecommender}

recs = []
for alg_id, rec_class in recommenders.iteritems():
    recs.extend(generate_recommendations(rec_class, train_df, alg_id, current=current, featured=featured))


df = pd.DataFrame(recs)
df.index.rename("id", inplace=True)
df.to_sql("recs_list", connection, if_exists="append", flavor="mysql")