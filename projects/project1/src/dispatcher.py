from sklearn import ensemble


MODELS = {
    "forest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extra": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2)
}