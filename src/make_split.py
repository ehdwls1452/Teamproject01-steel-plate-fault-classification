from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE, TARGET_COLS


# 프로젝트 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "Faults.csv"
SPLIT_DIR = BASE_DIR / "data" / "split"


# 1. 원본 데이터 불러오기
df = pd.read_csv(DATA_PATH)


# 2. Target 구조 확인
target_sum = df[TARGET_COLS].sum(axis=1)

if not (target_sum == 1).all():
    raise ValueError("Target이 정확히 하나만 1인 행이 존재하지 않습니다.")


# 3. 단일 multiclass target 생성
target = df[TARGET_COLS].idxmax(axis=1)


# 4. Stratified Train/Test Split
train_idx, test_idx = train_test_split(
    df.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=target
)


# 5. 저장 폴더 생성
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# 6. index 저장
pd.DataFrame({"index": train_idx}).to_csv(
    SPLIT_DIR / "train_idx.csv",
    index=False
)

pd.DataFrame({"index": test_idx}).to_csv(
    SPLIT_DIR / "test_idx.csv",
    index=False
)


# 7. 결과 확인
print(f"전체 데이터: {len(df)}")
print(f"Train: {len(train_idx)}")
print(f"Test: {len(test_idx)}")

print("\n[Train 클래스 분포]")
print(target.loc[train_idx].value_counts().sort_index())

print("\n[Test 클래스 분포]")
print(target.loc[test_idx].value_counts().sort_index())