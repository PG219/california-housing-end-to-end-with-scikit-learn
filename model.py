"""
California Housing, End to End with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_housing
def load_housing():
    # TODO: Download housing.tgz once into tempfile.gettempdir() and read housing/housing.csv from it.
    url="https://github.com/ageron/data/raw/main/housing.tgz"

    path = os.path.join(tempfile.gettempdir(), "housing.tgz")

    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with tarfile.open(path) as housing_tar:
        csv_file = housing_tar.extractfile("housing/housing.csv")


        return pd.read_csv(csv_file)

# Step 2 - income_categories
def income_categories(df):
    # TODO: pd.cut median_income with edges [0, 1.5, 3, 4.5, 6, inf] and labels 1..5; return an int Series.

    return pd.cut(df["median_income"], bins=[0.0, 1.5,3.0, 4.5, 6.0, np.inf], labels=[1,2,3,4,5]).astype(int)

# Step 3 - stratified_split
from sklearn.model_selection import train_test_split
def stratified_split(df, test_size=0.2, random_state=42):
    # TODO: train_test_split stratified on income_categories(df); return (train_set, test_set).
    strata = income_categories(df)
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state, stratify = strata)


    return train_set, test_set

# Step 4 - explore_correlations
def explore_correlations(df):
    # TODO: Pearson correlation of every numeric column with median_house_value, sorted descending, target excluded.

    corr_matrix = df.corr(numeric_only = True)
    return corr_matrix['median_house_value'].drop("median_house_value").sort_values(ascending=False)

# Step 5 - add_ratio_features
def add_ratio_features(df):
    # TODO: Return a copy with rooms_per_house, bedrooms_ratio and people_per_house columns added.
    df_copy = df.copy()
    df_copy["rooms_per_house"] = df_copy["total_rooms"]/df_copy["households"]

    df_copy["bedrooms_ratio"] = df_copy["total_bedrooms"]/df_copy["total_rooms"]
    df_copy["people_per_house"] = df_copy["population"]/df_copy["households"]


    return df_copy

# Step 6 - split_features_labels
def split_features_labels(df):
    X = df.drop("median_house_value",axis=1)

    y = df["median_house_value"].copy()

    return X,y

# Step 7 - ClusterSimilarity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state


    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init = 10, random_state=self.random_state
        )

        self.kmeans_.fit(X,sample_weight=sample_weight)
        return self


    def transform(self, X):
        return rbf_kernel(X,self.kmeans_.cluster_centers_, gamma=self.gamma)


    def get_feature_names_out(self, names=None):
        # TODO: ["Cluster 0 similarity", ...]
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# Step 8 - numeric_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
def numeric_pipeline():
    # TODO: make_pipeline(SimpleImputer(median), StandardScaler())
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

# Step 9 - categorical_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def categorical_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="most_frequent",missing_values=None),
        OneHotEncoder(handle_unknown="ignore")
    )

# Step 10 - build_preprocessing
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

def build_preprocessing(n_clusters=10, gamma=1.0, random_state=42):
    # Log pipeline for heavy-tailed numerical features
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler()
    )
    
    # Columns for log transformation
    log_cols = ["total_bedrooms", "total_rooms", "population", "households", "median_income"]
    
    # Full preprocessing ColumnTransformer
    preprocessing = ColumnTransformer(
        transformers=[
            ("log", log_pipeline, log_cols),
            ("geo", ClusterSimilarity(n_clusters=n_clusters, gamma=gamma, random_state=random_state), ["latitude", "longitude"]),
            ("cat", categorical_pipeline(), ["ocean_proximity"]),
        ],
        remainder=numeric_pipeline()
    )
    
    return preprocessing

# Step 11 - rmse
def rmse(y_true, y_pred):
    # TODO: sqrt(mean((y_true - y_pred)^2)) as a float.
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype = float)


    return float(np.sqrt(np.mean((y_true-y_pred)**2)))

# Step 12 - dummy_baseline_rmse
from sklearn.dummy import DummyRegressor
def dummy_baseline_rmse(X, y):
    dummy = DummyRegressor(strategy="mean")

    dummy.fit(X,y)

    y_pred = dummy.predict(X)

    return rmse(y,y_pred)

# Step 13 - cross_val_rmse
from sklearn.model_selection import cross_val_score
def cross_val_rmse(model, X, y, cv=3):
    # TODO: cross_val_score with neg_root_mean_squared_error; return {'scores': [...], 'mean': ..., 'std': ...}.
    scores = -cross_val_score(model,X,y,cv=cv, scoring="neg_root_mean_squared_error")

    return{
        "scores": scores.tolist(),
        "mean": float(scores.mean()),
        "std": float(scores.std())
    }

# Step 14 - linear_model
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression



def linear_model(preprocessing):
    # TODO: make_pipeline(preprocessing, LinearRegression())
    return make_pipeline(preprocessing, LinearRegression())

# Step 15 - forest_model (not yet solved)
# TODO: implement

# Step 16 - random_search (not yet solved)
# TODO: implement

# Step 17 - test_rmse (not yet solved)
# TODO: implement

# Step 18 - bootstrap_rmse_ci (not yet solved)
# TODO: implement

# Step 19 - feature_importances (not yet solved)
# TODO: implement

# Step 20 - worst_errors (not yet solved)
# TODO: implement

# Step 21 - save_and_reload (not yet solved)
# TODO: implement

# Step 22 - predict_new (not yet solved)
# TODO: implement

