import pandas as pd

from src.data_processing import (
    create_rfm_features,
    merge_risk_label
)


def test_create_rfm_features():

    data = {
        "CustomerId": [1, 1, 2],
        "TransactionId": [101, 102, 103],
        "Amount": [100, 200, 300],
        "TransactionStartTime": pd.to_datetime([
            "2025-01-01",
            "2025-01-02",
            "2025-01-03"
        ])
    }

    df = pd.DataFrame(data)

    rfm = create_rfm_features(df)

    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns


def test_merge_risk_label():

    df = pd.DataFrame({
        "CustomerId": [1, 2]
    })

    rfm = pd.DataFrame({
        "CustomerId": [1, 2],
        "is_high_risk": [1, 0]
    })

    merged = merge_risk_label(
        df,
        rfm
    )

    assert "is_high_risk" in merged.columns