from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE


# 프로젝트 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "Faults_processed.csv"
SPLIT_DIR = BASE_DIR / "data" / "split"


# 1. 전처리 완료 데이터 불러오기
df = pd.read_csv(DATA_PATH)


# 2. Target 확인
if "target" not in df.columns:
    raise ValueError("전처리 데이터에 'target' 컬럼이 없습니다.")

target = df["target"]


# 3. Stratified Train/Test Split
train_idx, test_idx = train_test_split(
    df.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=target
)


# 4. 저장 폴더 생성
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# 5. 공통 Train/Test index 저장
pd.DataFrame({"index": train_idx}).to_csv(
    SPLIT_DIR / "train_idx.csv",
    index=False
)

pd.DataFrame({"index": test_idx}).to_csv(
    SPLIT_DIR / "test_idx.csv",
    index=False
)


# 6. 결과 확인
print(f"전체 데이터: {len(df)}")
print(f"Train: {len(train_idx)}")
print(f"Test: {len(test_idx)}")
print(f"Train + Test: {len(train_idx) + len(test_idx)}")

print("\n[Train 클래스 분포]")
print(target.loc[train_idx].value_counts().sort_index())

print("\n[Test 클래스 분포]")
print(target.loc[test_idx].value_counts().sort_index())

print(
    "\nTrain/Test 중복:",
    len(set(train_idx) & set(test_idx))
)

print(
    "전체 행 포함:",
    set(train_idx) | set(test_idx) == set(df.index)
)