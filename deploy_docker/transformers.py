from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TransformerDates(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        date_column = pd.to_datetime(X["pickup_datetime"])
        date_df = pd.DataFrame()
        date_df['weekday'] = date_column.dt.weekday
        date_df['hour'] = date_column.dt.hour
        return date_df

class TransformerDistance(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_init = X[["pickup_latitude", "pickup_longitude"]].to_numpy()
        X_final = X[["dropoff_latitude", "dropoff_longitude"]].to_numpy()
        distance = self.haversine_distance(X_init, X_final)
        distance_df = pd.DataFrame()
        distance_df["distance"] = distance
        return distance_df
    
    def haversine_distance(self, X_init, X_final):
        # From decimals to radians
        X_init = np.radians(X_init)
        X_final = np.radians(X_final)
        # Haversine formula
        dlat = X_final[:, 0] - X_init[:, 0] 
        dlon = X_final[:, 1] - X_init[:, 1]
        a = np.sin(dlat / 2) ** 2 + np.cos(X_init[:, 0]) * np.cos(X_final[:, 0]) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
        return c * r

class TransformerSpeed(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        X_init = X[["pickup_latitude", "pickup_longitude"]].to_numpy()
        X_final = X[["dropoff_latitude", "dropoff_longitude"]].to_numpy()

        # Distancia de Haversine
        distancia = self.haversine_distance(X_init=X_init, X_final=X_final)
        
        speed_df = pd.DataFrame()
        time_in_hrs = y.to_numpy() / 3600
        speed_df["speed"] = distancia / time_in_hrs
        speed_df["pickup_borough"] = X["pickup_borough"]
        speed_borough = speed_df.groupby("pickup_borough")["speed"].mean()
        self.speed_borough = speed_borough.to_dict()
        return self

    def transform(self, X, y=None):
        speed_df = pd.DataFrame()
        speed_df["speed"] = X["pickup_borough"].map(self.speed_borough)
        return speed_df
    
    def haversine_distance(self, X_init, X_final):
        # From decimals to radians
        X_init = np.radians(X_init)
        X_final = np.radians(X_final)
        # Haversine formula
        dlat = X_final[:, 0] - X_init[:, 0] 
        dlon = X_final[:, 1] - X_init[:, 1]
        a = np.sin(dlat / 2) ** 2 + np.cos(X_init[:, 0]) * np.cos(X_final[:, 0]) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
        return c * r




