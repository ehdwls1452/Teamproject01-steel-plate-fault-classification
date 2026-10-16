"""교차검증 공용 함수. notebooks/03_modeling.ipynb 에서 사용한다.

- cv_run       : 주어진 분할 방식으로 교차검증을 돌려 fold별 점수와 CV 예측(out-of-fold)을 반환
- summarize    : cv_run 결과를 한 줄 요약 (fold 평균 ± 표준편차, CV 예측 기준 점수)
- per_class_f1 : CV 예측 기준 유형별 F1

보고 기준
- fold 평균 : fold마다 점수를 계산해 평균 (README의 공식 보고 방식). Accuracy·Precision·Recall도 같은 방식
- CV 예측 Macro F1   : 모든 fold의 검증 예측을 모아 한 번에 계산
- 표준편차는 모집단 표준편차(ddof=0)
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def cv_run(model, X, y, cv, groups=None):
    """교차검증을 실행한다.

    model  : 학습 전 모델 (fold마다 clone 해서 새로 학습)
    X, y   : 입력 변수와 정수 라벨
    cv     : StratifiedKFold, StratifiedGroupKFold 등 분할기
    groups : 그룹 분할기를 쓸 때의 그룹 라벨

    반환값 : {"pred": CV 예측(정수 라벨), "folds": fold별 점수 DataFrame}
    """
    y = np.asarray(y)
    pred = np.empty(len(y), dtype=y.dtype)
    rows = []
    for k, (tr, va) in enumerate(cv.split(X, y, groups)):
        m = clone(model).fit(_rows(X, tr), y[tr])
        p = np.asarray(m.predict(_rows(X, va))).ravel()
        pred[va] = p
        rows.append({
            "fold": k + 1,
            "n_val": len(va),
            "f1_macro": f1_score(y[va], p, average="macro"),
            "accuracy": accuracy_score(y[va], p),
            "precision_macro": precision_score(y[va], p, average="macro", zero_division=0),
            "recall_macro": recall_score(y[va], p, average="macro", zero_division=0),
        })
    return {"pred": pred, "folds": pd.DataFrame(rows).set_index("fold")}


def summarize(name, res, y):
    """cv_run 결과를 표 한 줄로 요약한다."""
    y = np.asarray(y)
    folds = res["folds"]
    f = folds["f1_macro"]
    return {
        "실험": name,
        "Accuracy": folds["accuracy"].mean(),           # fold 평균
        "Macro_P": folds["precision_macro"].mean(),     # fold 평균
        "Macro_R": folds["recall_macro"].mean(),        # fold 평균
        "MacroF1_fold평균": f.mean(),
        "MacroF1_std": f.std(ddof=0),
        "MacroF1_CV예측": f1_score(y, res["pred"], average="macro"),
    }


def per_class_f1(res, y, classes):
    """CV 예측 기준 유형별 F1. classes 는 정수 라벨 순서와 같은 이름 목록."""
    y = np.asarray(y)
    scores = f1_score(y, res["pred"], labels=list(range(len(classes))), average=None, zero_division=0)
    return pd.Series(scores, index=list(classes))
