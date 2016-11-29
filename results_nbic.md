# Project Results
Two recommender systems were evaluated locally by rolling crossvalidation and in production environment via A/B Testing.
All recommeders were integrated into web service back-end via shared database table that is filled with personalized recommendations daily.

Local validation results (Evaluate.ipynb) show that FMachine recommender predicts next purchase better than all other recommenders.

A/B testing results show that recommender based on SVD clusterization results in highest revenue per unique session.
