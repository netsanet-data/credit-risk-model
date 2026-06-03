import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.cluster import KMeans


def load_data(path):
    """
    Load raw transaction data
    """
    return pd.read_csv(path)


def create_aggregate_features(df):

    agg_df = df.groupby("CustomerId").agg(
        Total_Transaction_Amount=("Amount", "sum"),
        Average_Transaction_Amount=("Amount", "mean"),
        Transaction_Count=("Amount", "count"),
        Std_Transaction_Amount=("Amount", "std")
    ).reset_index()

    return agg_df


def extract_datetime_features(df):

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["Transaction_Hour"] = (
        df["TransactionStartTime"].dt.hour
    )

    df["Transaction_Day"] = (
        df["TransactionStartTime"].dt.day
    )

    df["Transaction_Month"] = (
        df["TransactionStartTime"].dt.month
    )

    df["Transaction_Year"] = (
        df["TransactionStartTime"].dt.year
    )

    return df


def calculate_rfm(df):

    snapshot_date = (
        pd.to_datetime(df["TransactionStartTime"])
        .max()
        + pd.Timedelta(days=1)
    )

    rfm = df.groupby("CustomerId").agg(
        Recency=(
            "TransactionStartTime",
            lambda x: (
                snapshot_date -
                pd.to_datetime(x.max())
            ).days
        ),
        Frequency=(
            "TransactionId",
            "count"
        ),
        Monetary=(
            "Amount",
            "sum"
        )
    ).reset_index()

    return rfm


def create_proxy_target(df):

    rfm = calculate_rfm(df)

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    rfm["Cluster"] = kmeans.fit_predict(
        rfm_scaled
    )

    cluster_summary = (
        rfm.groupby("Cluster")
        [["Recency", "Frequency", "Monetary"]]
        .mean()
    )

    high_risk_cluster = (
        cluster_summary["Frequency"]
        .idxmin()
    )

    rfm["is_high_risk"] = np.where(
        rfm["Cluster"] == high_risk_cluster,
        1,
        0
    )

    return rfm[
        ["CustomerId", "is_high_risk"]
    ]


def build_pipeline(df):

    categorical_columns = [
        "ProductCategory",
        "ChannelId",
        "PricingStrategy"
    ]

    numerical_columns = [
        "Amount",
        "Value"
    ]

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numerical_columns
            ),
            (
                "cat",
                categorical_transformer,
                categorical_columns
            )
        ]
    )

    return preprocessor


def create_rfm_features(df):

    snapshot_date = df["TransactionStartTime"].max()

    rfm = df.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
        "TransactionId": "count",
        "Amount": "sum"
    })

    rfm.columns = [
        "Recency",
        "Frequency",
        "Monetary"
    ]

    return rfm


def create_high_risk_label(rfm):

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    rfm["Cluster"] = kmeans.fit_predict(
        rfm_scaled
    )

    cluster_summary = rfm.groupby(
        "Cluster"
    )[["Recency", "Frequency", "Monetary"]].mean()

    high_risk_cluster = cluster_summary[
        "Frequency"
    ].idxmin()

    rfm["is_high_risk"] = (
        rfm["Cluster"] == high_risk_cluster
    ).astype(int)

    return rfm


def merge_risk_label(df, rfm):

    if "CustomerId" not in rfm.columns:
        rfm = rfm.reset_index()

    df = df.merge(
        rfm[
            ["CustomerId", "is_high_risk"]
        ],
        on="CustomerId",
        how="left"
    )

    return df
