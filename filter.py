

def recommend(user_id, recommender, count, whitelist=None, blacklist=None):
    """Filter function that yields required number of predictions while handling white- and black- lists correctly"""
    if whitelist is None:
        print "Must provide whitelist!"
        raise ValueError

    if blacklist is None:
        blacklist = set()

    allowed = whitelist.difference(blacklist)
    prediction = [i for i in recommender.predict(user_id) if i in allowed][:count]
    return prediction