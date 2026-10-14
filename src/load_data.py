from pathlib import Path

import pandas as pd


# 프로젝트 경로
BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "Faults_processed.csv"
TRAIN_IDX_PATH = BASE_DIR / "data" / "split" / "train_idx.csv"
TEST_IDX_PATH = BASE_DIR / "data" / "split" / "test_idx.csv"


def load_data():
    # 1. 전처리 완료 데이터 불러오기
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # 2. Feature / Target 분리
    X = df.drop(columns=["target"])
    y = df["target"]

    # 3. 공통 Train/Test index 불러오기
    train_idx = pd.read_csv(TRAIN_IDX_PATH)["index"]
    test_idx = pd.read_csv(TEST_IDX_PATH)["index"]

    # 4. 공통 index를 기준으로 Train/Test 분리
    X_train = X.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()

    y_train = y.loc[train_idx].copy()
    y_test = y.loc[test_idx].copy()

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    print("\nTrain Target 분포")
    print(y_train.value_counts())

    print("\nTest Target 분포")
    print(y_test.value_counts())