from sklearn.ensemble import RandomForestRegressor


def get_estimator():
    reg = RandomForestRegressor(
        n_estimators=20, max_leaf_nodes=2, random_state=61
    )
    return reg
