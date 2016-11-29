from filter import recommend
import pandas as pd
from pandas.tseries.offsets import DateOffset
import sklearn.externals.joblib as jl
from read_utils import train_test
import json

"""Multiple utility functions that provide wrappers for evaluaton procedures, including parallel evaluation"""


def top_n(actual, predicted):
    """Simplistic top_n metric - returns 1 if we guessed at least one purchase i our prediction"""
    for pred in predicted:
        if pred in actual:
            return 1.0
    return 0.0


def evaluate(rec, test, metric=top_n, count=6, whitelist=None, blacklist=None):
    """Returns average top_n for all users testing on transactions
    from test dataframe and predicting via pre-trained recommeder rec"""
    acc = []
    for user in test["user_id"].unique():
        user_actual = list(test[test["user_id"] == user]["item_set_id"])
        if user_actual != []:
            acc.append(metric(user_actual, recommend(user, rec, count, whitelist=whitelist, blacklist=blacklist)))
    return sum(acc) / len(acc)


def rolling_crossval(input_file_path, RecommenderClass, step=DateOffset(months=0, days=7),
                     fold_size=DateOffset(months=1), start_offset=DateOffset(years=1, months=0, days=0)):
    """Evaluates given recommeder class on each time interval in parallel,
     starting on current date - start_offset with step fold_size"""
    df = pd.read_csv(input_file_path, encoding="utf-8", low_memory=False)
    df['order_date'] = pd.to_datetime(df['order_date'])
    cur_time = df["order_date"].max()

    current_date = cur_time - start_offset + fold_size
    last_step_start = cur_time - fold_size

    current_dates = list()
    while current_date <= last_step_start:
        current_dates.append(current_date)
        current_date += step

    jl.Parallel(n_jobs=-1)(jl.delayed(calc)(fold_size, input_file_path, current_date,
                                                RecommenderClass) for current_date in current_dates)


def calc(fold_size, input_file_path, current_date, RecommenderClass):
    """Evaluates RecommenderClass on given current_date, using all transactions that happened after
    current_date - fold_size as test set"""
    train, test = train_test(input_file_path, current_date, test_size=fold_size)

    print RecommenderClass, len(train), len(test)
    current_sets = set(train[train["order_date"] > (current_date - fold_size - DateOffset(days=7))]["item_set_id"].unique())
    rec = RecommenderClass(train)
    accuracy = evaluate(rec, test, whitelist=current_sets)
    train_end_date = train["order_date"].max()

    with open("crossval_log", "a") as outfile:
        outfile.write(json.dumps({"RecommenderClass": str(RecommenderClass),
                                  "train_end_date": str(train_end_date),
                                  "accuracy": accuracy}) + "\n")