"""CV 평가 공용 함수.

한 번의 CV 실행으로 두 가지 방식의 지표를 함께 계산한다.
  - fold 평균 ± 표준편차 : fold별 지표를 계산한 뒤 평균 (안정성 확인용)
  - CV 예측 기준(pooled)  : 모든 Validation fold 예측을 모아 한 번에 계산
                           (evaluate.log_result, 유형별 F1, 혼동행렬에 사용)
Sampling·Scaling은 model(Pipeline) 안에 넣어 전달하면 각 Training fold에서만 적용된다.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)


def cv_run(model, X, y, cv, groups=None):
    """
    model  : 학습되지 않은 estimator 또는 Pipeline
    X, y   : Train 데이터 (y는 정수 인코딩된 배열 권장)
    cv     : StratifiedKFold / StratifiedGroupKFold 등
    groups : Group CV일 때만 전달

    반환: {"pred": CV 예측 배열, "folds": fold별 지표 DataFrame}
    """
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    pred = np.empty_like(y)
    rows = []

    for k, (tr, va) in enumerate(cv.split(X, y, groups)):
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[va])
        pred[va] = p
        rows.append({
            "fold": k,
            "accuracy": accuracy_score(y[va], p),
            "precision_macro": precision_score(y[va], p, average="macro", zero_division=0),
            "recall_macro": recall_score(y[va], p, average="macro", zero_division=0),
            "f1_macro": f1_score(y[va], p, average="macro"),
        })

    return {"pred": pred, "folds": pd.DataFrame(rows)}


def summarize(name, result, y):
    """cv_run 결과를 비교표 한 행으로 요약한다."""
    f = result["folds"]
    y = np.asarray(y)
    return {
        "실험": name,
        "Accuracy": f["accuracy"].mean(),
        "Macro_P": f["precision_macro"].mean(),
        "Macro_R": f["recall_macro"].mean(),
        "MacroF1_fold평균": f["f1_macro"].mean(),
        "MacroF1_std": f["f1_macro"].std(ddof=0),
        "MacroF1_CV예측": f1_score(y, result["pred"], average="macro"),
    }


def per_class_f1(result, y, class_names):
    """CV 예측 기준 유형별 F1 (Series, index=유형명)."""
    return pd.Series(
        f1_score(np.asarray(y), result["pred"], average=None,
                 labels=range(len(class_names))),
        index=class_names,
    )
