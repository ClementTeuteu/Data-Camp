from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def get_estimator():
    base_transformer= Pipeline(steps=[
            ("imputer", IterativeImputer(max_iter=10, random_state=42)),  
            ("scaler", StandardScaler())  
        ])
    
    reg = Pipeline(
        steps=[("transformer", base_transformer), ("regressor", RandomForestRegressor(
            n_estimators=20, max_leaf_nodes=2, random_state=61))]
    )
    return reg
