"""
모델링 공통 유틸 (Baseline·불균형 처리·튜닝·최종 평가 노트북에서 재사용)

- 팀 공통 config.py(RANDOM_STATE, CV, SCORING)를 그대로 사용하며 config.py는 수정하지 않는다.
- 데이터로부터 학습하는 전처리(결측 대체, Scaling)는 build_preprocessor()로 만들어
  모델 Pipeline 안에 넣는다 → CV의 각 Training fold에서만 fit 된다.
- evaluate_cv(): Train 내부 5-fold CV만 사용한다. Test 데이터는 이 모듈에서 다루지 않는다.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler

from config import CV, RANDOM_STATE, SCORING

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
# 혼동행렬·리포트에서 팀 전체가 동일하게 쓰는 클래스 순서 (건수 내림차순)
CLASS_ORDER = ["Other_Faults", "Bumps", "K_Scratch", "Z_Scratch", "Pastry", "Stains", "Dirtiness"]
MINORITY_CLASSES = ["Pastry", "Stains", "Dirtiness"]

# 이산형 변수: Scaling 제외, SMOTENC categorical_features 후보
DISCRETE_FEATURES = ["TypeOfSteel_A300", "Outside_Global_Index"]

# Train 기준 |skew| > 2 → 전처리 변경 실험(log1p) 후보
SKEWED_FEATURES = ["Pixels_Areas", "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity",
                   "Outside_X_Index", "Y_Minimum", "Y_Maximum", "Steel_Plate_Thickness"]

# 비교표 양식(Precision/Recall 열)과 확률 기반 보조 지표를 포함한 평가 지표
SCORING_EXTENDED = {
    **SCORING,
    "macro_precision": "precision_macro",
    "macro_recall": "recall_macro",
    "macro_roc_auc_ovr": "roc_auc_ovr",
}

# 평가 기준에서 정의한 "중점적으로 볼 오분류" (실제 → 예측)
KEY_ERRORS = {
    "소수 유형 → Other_Faults·Bumps": ([*MINORITY_CLASSES], ["Other_Faults", "Bumps"]),
    "Other_Faults → 소수 유형 (오경보)": (["Other_Faults"], [*MINORITY_CLASSES]),
    "Bumps ↔ Other_Faults": (["Bumps", "Other_Faults"], ["Other_Faults", "Bumps"]),
    "K_Scratch ↔ Z_Scratch": (["K_Scratch", "Z_Scratch"], ["Z_Scratch", "K_Scratch"]),
}


# ---------------------------------------------------------------------------
# 전처리
# ---------------------------------------------------------------------------
def _log1p_factory(idx):
    def _log1p_cols(X):
        X = np.asarray(X, dtype=float).copy()
        X[:, idx] = np.log1p(np.clip(X[:, idx], 0, None))
        return X
    return _log1p_cols


def build_preprocessor(feature_names, scale: bool = False, log_skewed: bool = False) -> ColumnTransformer:
    """
    scale=False : Tree 계열 (결측 대체만, Scaling 미적용)
    scale=True  : Logistic Regression (연속형 StandardScaler, 이산형은 Scaling 제외)
    log_skewed  : 왜도 큰 변수 log1p 변환 (전처리 변경 실험용)
    현재 결측 0건이지만 신규 데이터 대비로 SimpleImputer 유지.
    """
    feature_names = list(feature_names)
    discrete = [c for c in DISCRETE_FEATURES if c in feature_names]
    continuous = [c for c in feature_names if c not in discrete]

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if log_skewed:
        idx = [continuous.index(c) for c in SKEWED_FEATURES if c in continuous]
        steps.append(("log1p", FunctionTransformer(_log1p_factory(idx), feature_names_out="one-to-one")))
    if scale:
        steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        [("cont", Pipeline(steps), continuous),
         ("disc", SimpleImputer(strategy="most_frequent"), discrete)],
        verbose_feature_names_out=False,
    )


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """
    문자열 라벨을 정수로 바꿔 학습하고, 예측은 다시 문자열로 돌려주는 래퍼.
    XGBoost처럼 정수 라벨만 받는 모델을 sklearn 모델과 동일한 방식(문자열 target)으로 평가하기 위해 사용.
    sample_weight 등 fit 인자는 그대로 내부 모델에 전달된다.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.estimator_ = clone(self.estimator).fit(X, self.le_.transform(y), **fit_params)
        return self

    def predict(self, X):
        return self.le_.inverse_transform(np.asarray(self.estimator_.predict(X)).astype(int).ravel())

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


def make_pipeline(model, feature_names, scale: bool = False, log_skewed: bool = False) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(feature_names, scale, log_skewed)), ("model", model)])


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def evaluate_cv(name: str, estimator, X, y, cv=CV, scoring=SCORING_EXTENDED, fit_params=None, n_jobs=None):
    """
    Train 내부 Stratified 5-fold CV.
    반환:
      folds  : fold별 지표 (train/test)
      oof    : out-of-fold 예측 (Train 전체 1건당 1개 예측 → 클래스별 지표·혼동행렬용)
      n_conv_warn : ConvergenceWarning 발생 횟수 (LR 수렴 점검)
    """
    fit_params = fit_params or {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        t0 = time.perf_counter()
        res = cross_validate(estimator, X, y, cv=cv, scoring=scoring, return_train_score=True,
                             params=fit_params, n_jobs=n_jobs)
        cv_sec = time.perf_counter() - t0
        oof = cross_val_predict(estimator, X, y, cv=cv, params=fit_params, n_jobs=n_jobs)
    n_conv = sum(issubclass(x.category, ConvergenceWarning) for x in w)

    folds = pd.DataFrame({k.replace("test_", "val_"): v for k, v in res.items()})
    folds.insert(0, "fold", range(1, len(folds) + 1))
    folds.insert(0, "model", name)
    return {"name": name, "folds": folds, "oof": pd.Series(oof, index=y.index, name=name),
            "n_conv_warn": n_conv, "cv_sec": cv_sec}


def summarize_cv(results: list[dict]) -> pd.DataFrame:
    """모델별 CV 평균 ± 표준편차 표 (Validation fold 기준) + Train-Val 차이(과적합 지표)."""
    rows = []
    for r in results:
        f = r["folds"]
        row = {"model": r["name"]}
        for m in SCORING_EXTENDED:
            row[f"{m}_mean"] = f[f"val_{m}"].mean()
            row[f"{m}_std"] = f[f"val_{m}"].std(ddof=0)
        row["train_macro_f1"] = f["train_macro_f1"].mean()
        row["overfit_gap_macro_f1"] = row["train_macro_f1"] - row["macro_f1_mean"]
        row["fit_time_sec"] = f["fit_time"].mean()
        row["convergence_warnings"] = r["n_conv_warn"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def per_class_report(y_true, y_pred, labels=CLASS_ORDER) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=labels)


def confusion_df(y_true, y_pred, labels=CLASS_ORDER, normalize=None) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return pd.DataFrame(cm, index=pd.Index(labels, name="실제"), columns=pd.Index(labels, name="예측"))


def key_error_counts(y_true, y_pred) -> pd.Series:
    """평가 기준에서 정의한 주요 오분류 유형별 건수."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    out = {}
    for k, (src, dst) in KEY_ERRORS.items():
        if "↔" in k:  # 양방향 두 클래스 간 혼동
            a, b = src
            out[k] = int(((yt == a) & (yp == b)).sum() + ((yt == b) & (yp == a)).sum())
        else:
            out[k] = int((np.isin(yt, src) & np.isin(yp, dst)).sum())
    out["전체 오분류"] = int((yt != yp).sum())
    return pd.Series(out)


def oof_macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro")


# ---------------------------------------------------------------------------
# 클래스 불균형 처리 (가중치 / Sampling) — 모두 CV의 Training fold 내부에서만 계산·적용
# ---------------------------------------------------------------------------
def class_weight_dict(y, alpha: float = 1.0) -> dict:
    """
    클래스 가중치 w_c = (n_max / n_c) ** alpha  (최다 클래스 = 1)
      alpha=0   → 가중치 없음 (원래 분포 유지)
      alpha=0.5 → 부분 완화 (최대/최소 가중치 비 = sqrt(12.2) ≈ 3.5)
      alpha=1   → 균등화 (sklearn 'balanced'와 비율 동일: 클래스별 가중치 합이 같아짐)
    y는 Training fold의 라벨만 넘겨야 한다(누수 방지).
    """
    counts = pd.Series(y).value_counts()
    return {c: float((counts.max() / n) ** alpha) for c, n in counts.items()}


def sample_weight_from(y, alpha: float = 1.0) -> np.ndarray:
    w = class_weight_dict(y, alpha)
    return pd.Series(y).map(w).to_numpy()


class _IRStrategy:
    """Sampler용 목표 건수 함수 (pickle 가능, 출력 표기: 'IR=1', 'IR≤7')."""
    def __init__(self, ir: float, mode: str):
        self.ir, self.mode = ir, mode

    def __call__(self, y):
        c = pd.Series(y).value_counts()
        if self.mode == "under":
            cap = int(round(self.ir * c.min()))
            return {k: int(min(n, cap)) for k, n in c.items()}
        floor = int(round(c.max() / self.ir))
        return {k: int(max(n, floor)) for k, n in c.items()}

    def __repr__(self):
        return f"IR={self.ir:g}" if self.ir == 1 else f"IR≤{self.ir:g}"

    def __eq__(self, other):
        return isinstance(other, _IRStrategy) and (self.ir, self.mode) == (other.ir, other.mode)

    def __hash__(self):
        return hash((self.ir, self.mode))


def under_strategy(ir: float):
    """RandomUnderSampler용: 각 클래스를 min(n_c, ir × n_min)으로 축소 (최대/최소 비율 ≤ ir)."""
    return _IRStrategy(ir, "under")


def over_strategy(ir: float):
    """SMOTE/SMOTENC용: 각 클래스를 max(n_c, n_max / ir)로 증강 (최대/최소 비율 ≤ ir)."""
    return _IRStrategy(ir, "over")


def target_counts(y, method: str, ir: float | None = None, alpha: float | None = None) -> pd.Series:
    """설명용: 주어진 라벨 분포에서 처리 후 클래스별 목표 건수(Sampling) 또는 유효 가중 건수(가중치)."""
    c = pd.Series(y).value_counts().reindex(CLASS_ORDER)
    if method == "under":
        return pd.Series(under_strategy(ir)(y)).reindex(CLASS_ORDER)
    if method == "over":
        return pd.Series(over_strategy(ir)(y)).reindex(CLASS_ORDER)
    if method == "weight":
        w = pd.Series(class_weight_dict(y, alpha)).reindex(CLASS_ORDER)
        return (c * w).round(1)
    return c


def _fit_one_fold(build_estimator, X, y, tr, va, k, weight_alpha, weight_mode, labels):
    """fold 1개 학습·평가 (병렬 실행 단위). 가중치·Sampling은 이 fold의 Training 부분에만 적용."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        est = build_estimator()
        fit_kw = {}
        if weight_alpha is not None:
            if weight_mode == "class_weight":
                est.set_params(model__class_weight=class_weight_dict(ytr, weight_alpha))
            else:
                fit_kw["model__sample_weight"] = sample_weight_from(ytr, weight_alpha)
        t0 = time.perf_counter()
        est.fit(Xtr, ytr, **fit_kw)
        fit_sec = time.perf_counter() - t0
        pva, ptr, proba = est.predict(Xva), est.predict(Xtr), est.predict_proba(Xva)
    n_conv = sum(issubclass(x.category, ConvergenceWarning) for x in w)

    # Sampling 후 학습 건수: 샘플러의 목표 건수 함수로 계산 (재샘플링 없이)
    n_after = len(ytr)
    steps = getattr(est, "named_steps", {})
    sampler = steps.get("sampler") if hasattr(steps, "get") else None
    if sampler is not None and not isinstance(sampler, str) and callable(getattr(sampler, "sampling_strategy", None)):
        n_after = int(sum(sampler.sampling_strategy(ytr).values()))

    row = {"fold": k, "fit_time": fit_sec, "n_train_after": n_after,
           "val_macro_f1": f1_score(yva, pva, average="macro"),
           "val_weighted_f1": f1_score(yva, pva, average="weighted"),
           "val_accuracy": accuracy_score(yva, pva),
           "val_macro_precision": precision_score(yva, pva, average="macro", zero_division=0),
           "val_macro_recall": recall_score(yva, pva, average="macro"),
           "val_macro_roc_auc_ovr": roc_auc_score(yva, proba, multi_class="ovr", labels=list(est.classes_)),
           "train_macro_f1": f1_score(ytr, ptr, average="macro")}
    rep = per_class_report(yva, pva, labels)
    rep["fold"] = k
    return {"row": row, "rep": rep, "va": va, "pred": pva, "n_conv": n_conv}


def _collect(name, y, parts):
    oof = pd.Series(index=y.index, dtype=object, name=name)
    for p in parts:
        oof.iloc[p["va"]] = p["pred"]
    folds = pd.DataFrame([{"model": name, **p["row"]} for p in parts])
    per_fold_cls = pd.concat([p["rep"] for p in parts]).rename_axis("class").reset_index().assign(model=name)
    return {"name": name, "folds": folds, "oof": oof, "per_fold_class": per_fold_cls,
            "n_conv_warn": sum(p["n_conv"] for p in parts), "cv_sec": folds["fit_time"].sum()}


def run_many(specs: dict, X, y, cv=CV, n_jobs: int = -1, labels=CLASS_ORDER, verbose: int = 0) -> dict:
    """
    여러 실험을 (실험 × fold) 단위로 한 번에 병렬 실행한다.
      specs = {key: dict(build=..., weight_alpha=None|float, weight_mode="sample_weight"|"class_weight")}
    → {key: run_cv와 같은 결과 dict}
    병렬 실행 시 모델 내부 n_jobs는 1로 두어야 CPU를 과다 점유하지 않는다.
    """
    from joblib import Parallel, delayed
    splits = list(cv.split(X, y))
    tasks = [(key, k, tr, va) for key in specs for k, (tr, va) in enumerate(splits, 1)]
    outs = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_fit_one_fold)(specs[key]["build"], X, y, tr, va, k, specs[key].get("weight_alpha"),
                               specs[key].get("weight_mode", "sample_weight"), labels)
        for key, k, tr, va in tasks)
    res = {}
    for key in specs:
        parts = [o for (kk, *_), o in zip(tasks, outs) if kk == key]
        name = key if isinstance(key, str) else " | ".join(map(str, key))
        res[key] = _collect(name, y, parts)
    return res


def run_cv(name: str, build_estimator, X, y, cv=CV, weight_alpha: float | None = None,
           weight_mode: str = "sample_weight", labels=CLASS_ORDER, n_jobs: int = 1):
    """
    수동 Stratified K-fold (fold당 1회 학습으로 지표·OOF·train 점수를 동시에 계산).
      build_estimator(): 매 fold 새 모델(Pipeline)을 반환하는 함수
      weight_alpha     : None이면 가중치 없음. 값이 있으면 Training fold 분포로 가중치 계산
      weight_mode      : "sample_weight" (GBM·XGBoost·LightGBM) / "class_weight" (LR·DT·RF)
    Sampling은 imblearn Pipeline 안에 있으므로 fit 시 Training fold에만 적용되고,
    Validation fold 예측(predict)에는 적용되지 않는다.
    """
    spec = {name: dict(build=build_estimator, weight_alpha=weight_alpha, weight_mode=weight_mode)}
    return run_many(spec, X, y, cv=cv, n_jobs=n_jobs, labels=labels)[name]


def make_sampling_pipeline(model, feature_names, sampler, scale: bool = False):
    """
    imblearn Pipeline: prep → sampler → model
    - prep(결측 대체/Scaling)은 Training fold로 fit
    - sampler는 fit 단계에서만 동작 (predict 시에는 건너뜀) → Validation/Test 분포 유지
    - ColumnTransformer 출력에서 이산형 변수는 항상 마지막 열들에 위치 → SMOTENC 범주 인덱스로 사용
    """
    from imblearn.pipeline import Pipeline as ImbPipeline
    return ImbPipeline([("prep", build_preprocessor(feature_names, scale)), ("sampler", sampler), ("model", model)])


def discrete_indices(feature_names) -> list[int]:
    """build_preprocessor 출력 기준 이산형 변수의 열 위치 (SMOTENC categorical_features)."""
    feature_names = list(feature_names)
    n_disc = len([c for c in DISCRETE_FEATURES if c in feature_names])
    n = len(feature_names)
    return list(range(n - n_disc, n))


def summarize_runs(results: list[dict], baseline_f1: dict | None = None) -> pd.DataFrame:
    rows = []
    for r in results:
        f = r["folds"]
        row = {"run": r["name"]}
        for m in ["macro_f1", "accuracy", "macro_precision", "macro_recall", "weighted_f1", "macro_roc_auc_ovr"]:
            row[f"{m}_mean"] = f[f"val_{m}"].mean()
            row[f"{m}_std"] = f[f"val_{m}"].std(ddof=0)
        row["train_macro_f1"] = f["train_macro_f1"].mean()
        row["overfit_gap"] = row["train_macro_f1"] - row["macro_f1_mean"]
        row["n_train_after"] = f["n_train_after"].mean()
        row["fit_time_sec"] = f["fit_time"].mean()
        row["convergence_warnings"] = r["n_conv_warn"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("run")


# ---------------------------------------------------------------------------
# 하이퍼파라미터 튜닝용: 클래스 가중치를 fit 내부에서 계산하는 래퍼
# ---------------------------------------------------------------------------
class BalancedWeightClassifier(ClassifierMixin, BaseEstimator):
    """
    fit(X, y)가 호출될 때 그 y(= CV의 Training fold)로 클래스 가중치 (n_max/n_c)^alpha를 계산해
    내부 모델의 sample_weight로 전달한다. alpha=0이면 가중치 없음.
    → GridSearchCV/RandomizedSearchCV 안에서도 Validation fold 정보 없이 가중치가 계산되고,
      alpha 자체를 튜닝 대상으로 삼을 수 있다. (class_weight 인자가 없는 GBM·XGBoost용)
    """

    def __init__(self, estimator, alpha: float = 1.0):
        self.estimator = estimator
        self.alpha = alpha

    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator)
        if self.alpha:
            fit_params = {**fit_params, "sample_weight": sample_weight_from(y, self.alpha)}
        self.estimator_.fit(X, y, **fit_params)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


# ---------------------------------------------------------------------------
# EDA·전처리: 원본 검증과 행 단위 전처리 (데이터 통계를 학습하지 않음 → 누수 없음)
# ---------------------------------------------------------------------------
TARGET_NAME = "target"
LABEL_RENAME = {"K_Scatch": "K_Scratch"}          # 원본 정답 열 이름 오타 수정 (팀 전처리와 동일)
DROPPED_FEATURES = ["TypeOfSteel_A400"]           # = 1 − TypeOfSteel_A300 (정의상 완전 중복)


def validate_raw(df: pd.DataFrame) -> None:
    """원본(data/raw/Faults.csv) 구조·품질·도메인 논리 검증. 문제가 있으면 AssertionError."""
    from config import TARGET_COLS
    assert df.shape == (1941, 34), f"예상과 다른 크기: {df.shape}"
    t = df[TARGET_COLS]
    assert t.isin([0, 1]).all().all(), "정답 열에 0/1 이외의 값이 있음"
    assert (t.sum(axis=1) == 1).all(), "정답 열의 행 합이 1이 아닌 행이 있음"
    assert df.isna().sum().sum() == 0, "결측값 존재"
    assert df.duplicated().sum() == 0, "전체 행 중복 존재"
    assert df.drop(columns=TARGET_COLS).duplicated().sum() == 0, "입력값 기준 중복 존재"
    assert (df["X_Minimum"] <= df["X_Maximum"]).all()
    assert (df["Y_Minimum"] <= df["Y_Maximum"]).all()
    assert (df["Minimum_of_Luminosity"] <= df["Maximum_of_Luminosity"]).all()
    assert (df["X_Maximum"] <= df["Length_of_Conveyer"]).all()
    assert (df["Pixels_Areas"] > 0).all()
    assert ((df["TypeOfSteel_A300"] + df["TypeOfSteel_A400"]) == 1).all(), "강종 표시가 상호배타적이지 않음"


def make_processed(df: pd.DataFrame) -> pd.DataFrame:
    """one-hot 정답 → 단일 target(오타 수정 포함), 정답 열 7개 제외, A400 제외. 행 순서·index 유지."""
    from config import TARGET_COLS
    out = df.copy()
    out[TARGET_NAME] = out[TARGET_COLS].idxmax(axis=1).replace(LABEL_RENAME)
    return out.drop(columns=TARGET_COLS + DROPPED_FEATURES)


# ---------------------------------------------------------------------------
# Feature Engineering: 행 단위 파생변수 (다른 행·Target 정보를 쓰지 않음 → 누수 없음)
# ---------------------------------------------------------------------------
FE_GROUPS = {
    "형상": ["compactness", "perimeter_ratio_log", "x_straightness"],
    "밝기": ["lum_range", "lum_contrast", "lum_mean_position"],
    "위치": ["x_center_rel", "x_center_dist"],
}


def add_features(X: pd.DataFrame, groups=("형상", "밝기", "위치")) -> pd.DataFrame:
    """
    형상: compactness          = (X_Perimeter + Y_Perimeter) / sqrt(Pixels_Areas)   둘레 대비 면적 (복잡한 윤곽일수록 큼)
          perimeter_ratio_log  = log(X_Perimeter / Y_Perimeter)                     가로/세로 윤곽 길이 비
          x_straightness       = X_Perimeter / (X 폭 + 1)                            X 방향 윤곽이 폭 대비 얼마나 구불구불한가
    밝기: lum_range            = Maximum − Minimum_of_Luminosity                      결함 내부 밝기 대비
          lum_contrast         = (Max − Min) / (Max + Min)                           상대 대비 (Michelson contrast)
          lum_mean_position    = (평균 밝기 − Min) / (Max − Min + 1)                   평균 밝기가 밝기 범위 중 어디에 있는가
    위치: x_center_rel         = X 중심 / Length_of_Conveyer                          강판 폭 방향 상대 위치
          x_center_dist        = |x_center_rel − 0.5|                                강판 중앙에서 떨어진 정도
    """
    out = X.copy()
    w = (X["X_Maximum"] - X["X_Minimum"]).clip(lower=0)
    if "형상" in groups:
        out["compactness"] = (X["X_Perimeter"] + X["Y_Perimeter"]) / np.sqrt(X["Pixels_Areas"])
        out["perimeter_ratio_log"] = np.log((X["X_Perimeter"] + 1) / (X["Y_Perimeter"] + 1))
        out["x_straightness"] = X["X_Perimeter"] / (w + 1)
    if "밝기" in groups:
        mx, mn = X["Maximum_of_Luminosity"], X["Minimum_of_Luminosity"]
        mean = X["Sum_of_Luminosity"] / X["Pixels_Areas"]
        out["lum_range"] = mx - mn
        out["lum_contrast"] = (mx - mn) / (mx + mn).replace(0, 1)
        out["lum_mean_position"] = (mean - mn) / (mx - mn + 1)
    if "위치" in groups:
        xc = (X["X_Minimum"] + X["X_Maximum"]) / 2 / X["Length_of_Conveyer"]
        out["x_center_rel"] = xc
        out["x_center_dist"] = (xc - 0.5).abs()
    # 이산형 변수(SMOTENC 범주)가 전처리 후 마지막 열에 오도록 순서 유지: build_preprocessor가 자동 처리
    return out


PLATE_KEYS = ["Length_of_Conveyer", "Steel_Plate_Thickness", "TypeOfSteel_A300"]


class PlateContextEncoder(TransformerMixin, BaseEstimator):
    """
    강판(생산 조건) 맥락 파생변수 — 피처 중요도 1·2위(Length_of_Conveyer, Steel_Plate_Thickness)와 강종을 결합.
      plate_defect_count : 같은 (컨베이어 길이, 두께, 강종) 조합의 결함 수 — fit에 들어온 Training fold에서만 계산 (Target 미사용)
      thickness_per_conveyer : Steel_Plate_Thickness / Length_of_Conveyer
    Validation/Test에 처음 보는 조합은 0으로 둔다.
    """

    def fit(self, X, y=None):
        key = X[PLATE_KEYS].astype(str).agg("_".join, axis=1)
        self.counts_ = key.value_counts().to_dict()
        return self

    def transform(self, X):
        out = X.copy()
        key = X[PLATE_KEYS].astype(str).agg("_".join, axis=1)
        out["plate_defect_count"] = key.map(self.counts_).fillna(0).astype(float)
        out["thickness_per_conveyer"] = X["Steel_Plate_Thickness"] / X["Length_of_Conveyer"]
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features) + ["plate_defect_count", "thickness_per_conveyer"])
