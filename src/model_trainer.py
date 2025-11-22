from xgboost import XGBClassifier
from data import load_dataset,split_data
from CONFIG import XGB_PARAMETERS
import joblib
import os

def train_model():
    if os.path.exists("models/xgb_model.pkl"):
        print("Model already exists.Loading it")
        model = joblib.load("models/xgb_model.pkl")

        df = load_dataset()
        X_train,X_test,y_train,y_test=split_data(df)
        return model, X_test, y_test
    
    df=load_dataset()
    X_train,X_test,y_train,y_test=split_data(df)

    # print(X_train.shape, X_test.shape)
    model=XGBClassifier(**XGB_PARAMETERS) #unpack dictionary items into separate strings
    model.fit(X_train,y_train)
    joblib.dump(model,"models/xgb_model.pkl")
    return model,X_test,y_test