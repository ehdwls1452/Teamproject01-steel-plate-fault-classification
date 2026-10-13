from pathlib import Path

import pandas as pd

from config import TARGET_COLS


# 프로젝트 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "Faults.csv"
TRAIN_IDX_PATH = BASE_DIR / "data" / "split" / "train_idx.csv"
TEST_IDX_PATH = BASE_DIR / "data" / "split" / "test_idx.csv"


def load_data():
    # 1. 원본 데이터 불러오기
    df = pd.read_csv(DATA_PATH)

    # 2. 단일 multiclass target 생성
    y = df[TARGET_COLS].idxmax(axis=1)

    # 3. 기존 Target 컬럼을 제외하여 Feature 생성
    X = df.drop(columns=TARGET_COLS)

    # 4. 공통 Train/Test index 불러오기
    train_idx = pd.read_csv(TRAIN_IDX_PATH)["index"]
    test_idx = pd.read_csv(TEST_IDX_PATH)["index"]

    # 5. 공통 index를 기준으로 Train/Test 분리
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