"""
대시보드 공통 도구 — 경로 · 화면 스타일 · 표/차트 그리기 · 데이터와 모델 불러오기 · 예측
(app.py 의 화면 코드가 짧고 읽기 쉽도록 반복되는 부분을 여기로 모았습니다.)
"""
from __future__ import annotations

import platform
import sys
import traceback
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.dashboard_data import CLASS_ORDER

# ──────────────────────────────────────────────────────────────────────────────
# 1. 경로 — src 폴더가 있는 곳을 프로젝트 루트로 봅니다
# ──────────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
ROOT = next((p for p in [HERE, HERE.parent, Path.cwd(), Path.cwd().parent] if (p / "src").is_dir()), HERE)
SRC_DIR, DATA_DIR, FIG_DIR = ROOT / "src", ROOT / "data", ROOT / "figures"
RES_DIR, MODEL_DIR = ROOT / "results", ROOT / "models"
MODEL_FILES = [MODEL_DIR / "Taegyeom_final_model_trained.pkl", MODEL_DIR / "Taegyeom_final_model.pkl"]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))       # 저장된 모델 안의 사용자 정의 클래스를 불러오려면 필요

# ──────────────────────────────────────────────────────────────────────────────
# 2. 화면 스타일
# ──────────────────────────────────────────────────────────────────────────────
BLUE, BLUE_LIGHT, ORANGE, RED, GREEN, GRAY = "#2a78d6", "#9dbde3", "#e07b2a", "#c8453b", "#3a9e5f", "#9aa1ab"
INK, INK2, MUTED, LINE = "#1d232b", "#4a525c", "#7c838c", "#e3e5e8"

CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
html, body, .stApp, .stMarkdown, p, li, h1, h2, h3, h4, label, input, textarea, button,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {{
    font-family: 'Pretendard', 'Apple SD Gothic Neo', 'Malgun Gothic', 'Noto Sans KR', sans-serif;
}}
.block-container {{ max-width: 1240px; padding-top: 2.2rem; padding-bottom: 3rem; }}
h1 {{ font-size: 1.75rem !important; font-weight: 700 !important; letter-spacing: -0.02em; margin-bottom: .2rem !important; }}
h2 {{ font-size: 1.25rem !important; font-weight: 700 !important; letter-spacing: -0.01em; margin-top: 1.6rem !important; }}
h3 {{ font-size: 1.05rem !important; font-weight: 650 !important; }}
p, li {{ line-height: 1.65; }}
.page-lead {{ color: {INK2}; font-size: 1rem; margin: 0 0 1.4rem 0; max-width: 62rem; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: .4rem 0 1.2rem 0; }}
.kpi {{ border: 1px solid {LINE}; border-radius: 10px; padding: 14px 16px 12px 16px; background: #ffffff; }}
.kpi .label {{ font-size: .84rem; color: #000000; font-weight: 500; margin-bottom: 4px; }}
.kpi .value {{ font-size: 1.6rem; font-weight: 700; color: #000000; font-variant-numeric: tabular-nums; line-height: 1.2; }}
.kpi .sub {{ font-size: .82rem; color: #000000; margin-top: 4px; font-variant-numeric: tabular-nums; }}
.kpi.accent {{ border-left: 4px solid {BLUE}; }}
.note {{ border-left: 3px solid {LINE}; padding: 4px 0 4px 14px; color: {INK2}; font-size: .93rem; margin: .6rem 0 1rem 0; }}
.fig-caption {{ color: {MUTED}; font-size: .85rem; margin: -.3rem 0 1.2rem 0; }}
.verdict {{ border-radius: 10px; padding: 14px 18px; margin: .4rem 0 1rem 0; font-size: 1rem; }}
.verdict.ok {{ background: #eaf3ec; border: 1px solid #bcdcc5; color: #1f5a33; }}
.verdict.check {{ background: #fdf0e6; border: 1px solid #f1c9a5; color: #7a3d0c; }}
.verdict b {{ font-size: 1.15rem; }}
</style>
"""


def apply_style():
    st.markdown(CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. 화면 조각 (제목 · 요약 카드 · 메모 · 그림 · 표 · 차트)
# ──────────────────────────────────────────────────────────────────────────────
def stretch(fn, *args, **kwargs):
    """Streamlit 버전에 상관없이 화면 폭에 맞춰 표시 (그림은 원본 비율 유지)."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def page_header(title: str, lead: str):
    st.title(title)
    st.markdown(f'<p class="page-lead">{lead}</p>', unsafe_allow_html=True)


def kpis(items: list[tuple[str, str, str]], accent_first: bool = False):
    """요약 카드 한 줄. items = [(제목, 큰 숫자, 보조 설명), ...]"""
    cards = "".join(
        f'<div class="kpi{" accent" if accent_first and i == 0 else ""}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
        for i, (label, value, sub) in enumerate(items))
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def caption(text: str):
    st.markdown(f'<p class="fig-caption">{text}</p>', unsafe_allow_html=True)


def figure(name: str, text: str | None = None):
    """figures 폴더의 그림을 원본 비율 그대로 보여줍니다."""
    path = FIG_DIR / name
    if not path.exists():
        st.warning(f"`figures/{name}` 파일을 찾지 못했습니다. 노트북에서 저장한 그림을 figures 폴더에 넣어 주세요.")
        return
    stretch(st.image, str(path))
    if text:
        caption(text)


def table(df: pd.DataFrame, fmt: dict[str, str] | None = None, height: int | None = None):
    """숫자는 오른쪽 정렬·자릿수 고정, 기록이 없는 값은 '—'로 표시."""
    df, cfg = df.copy(), {}
    for col, f in (fmt or {}).items():
        if col not in df.columns:
            continue
        if df[col].isna().any():
            df[col] = df[col].map(lambda v, f=f: "—" if pd.isna(v) else f % v)
            try:
                cfg[col] = st.column_config.TextColumn(col, alignment="right")
            except TypeError:                          # 오래된 Streamlit
                cfg[col] = st.column_config.TextColumn(col)
        else:
            cfg[col] = st.column_config.NumberColumn(col, format=f)
    kw = {"height": height} if height else {}
    stretch(st.dataframe, df, hide_index=True, column_config=cfg, **kw)


def chart(c, fit: bool = True):
    """공통 글꼴·격자 설정을 입혀 차트를 그립니다. fit=False 면 칸 크기를 고정(혼동행렬)."""
    c = (c.properties(padding={"left": 4, "right": 16, "top": 4, "bottom": 18})
          .configure_axis(labelFontSize=12, titleFontSize=12, labelColor=INK2, titleColor=INK2,
                          gridColor=LINE, domainColor="#c9ccd1", labelLimit=320)
          .configure_axisQuantitative(tickCount=8)
          .configure_legend(labelFontSize=12, titleFontSize=12, orient="top", labelLimit=320)
          .configure_view(strokeWidth=0))
    if fit:
        stretch(st.altair_chart, c)
    else:
        st.altair_chart(c)


def hbar(df, y, x, *, sort, color, tooltip, x_title=None, domain=None, start=None, bar=18, rounded=True,
         fmt=".4f", text_size=12, dx=5, group=None, group_sort=None, y_axis=alt.Undefined, x_axis=alt.Undefined):
    """가로 막대 + 막대 끝 숫자 라벨.
    domain=(최소, 최대): x축 범위 / start: 막대가 시작하는 값(x축이 0이 아닐 때) / group: 막대를 묶어 나란히 그릴 열"""
    offset = {"yOffset": alt.YOffset(f"{group}:N", sort=group_sort)} if group else {}
    base_x = {"x2": alt.datum(start)} if start is not None else {}
    mark = {"height": bar, "cornerRadiusEnd": 3} if rounded else {"height": bar}
    bars = alt.Chart(df).mark_bar(**mark).encode(
        y=alt.Y(f"{y}:N", sort=sort, title=None, axis=y_axis),
        x=alt.X(f"{x}:Q", scale=alt.Scale(domain=list(domain)) if domain else alt.Undefined, title=x_title, axis=x_axis),
        color=color, tooltip=tooltip, **base_x, **offset)
    labels = alt.Chart(df).mark_text(align="left", dx=dx, fontSize=text_size, color=INK2).encode(
        y=alt.Y(f"{y}:N", sort=sort), x=f"{x}:Q", text=alt.Text(f"{x}:Q", format=fmt), **offset)
    return bars + labels


def confusion_chart(cm: np.ndarray, title: str, mode: str = "행 기준 %"):
    """혼동행렬: 칸 색 = 행 기준 % 또는 건수, 오른쪽에 색 범위 막대."""
    rows = []
    for i, actual in enumerate(CLASS_ORDER):
        n = cm[i].sum()
        for j, pred in enumerate(CLASS_ORDER):
            rows.append({"실제": f"{actual} (n={n})", "예측": pred, "건수": int(cm[i, j]),
                         "행 비율(%)": cm[i, j] / n * 100 if n else 0, "정답": i == j})
    df = pd.DataFrame(rows)
    y_sort = [f"{a} (n={cm[i].sum()})" for i, a in enumerate(CLASS_ORDER)]
    val = "행 비율(%)" if mode == "행 기준 %" else "건수"
    vmax = 100 if mode == "행 기준 %" else int(cm.max())
    x_enc, y_enc = alt.X("예측:N", sort=CLASS_ORDER), alt.Y("실제:N", sort=y_sort)

    rect = alt.Chart(df).encode(
        x=alt.X("예측:N", sort=CLASS_ORDER, title="예측", axis=alt.Axis(labelAngle=-35, labelOverlap=False, labelAlign="right")),
        y=alt.Y("실제:N", sort=y_sort, title="실제"),
    ).mark_rect(stroke="white", strokeWidth=1.5).encode(
        color=alt.Color(f"{val}:Q",
                        scale=alt.Scale(range=["#f4f8fd", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"], domain=[0, vmax]),
                        # 색 범위 막대: 제목 + 막대 높이를 혼동행렬 높이(7칸 × 54px)에 맞춤
                        legend=alt.Legend(title=val, orient="right", direction="vertical",
                                          gradientLength=len(CLASS_ORDER) * 54 - 28, gradientThickness=14,
                                          titleOrient="top", titleFontSize=12, labelFontSize=11, tickCount=6, format=".0f")),
        tooltip=["실제", "예측", "건수", alt.Tooltip("행 비율(%):Q", format=".1f")])
    shown = df[df["건수"] > 0]
    txt_color = alt.condition(f"datum['{val}'] > {vmax * 0.55}", alt.value("white"), alt.value(INK))
    pct = alt.Chart(shown).mark_text(fontSize=13, dy=-8).encode(
        x=x_enc, y=y_enc, text=alt.Text("pct_label:N"), color=txt_color).transform_calculate(
        pct_label="format(datum['행 비율(%)'], '.0f') + '%'")
    cnt = alt.Chart(shown).mark_text(fontSize=11, dy=9).encode(
        x=x_enc, y=y_enc, text=alt.Text("cnt_label:N"), color=txt_color).transform_calculate(
        cnt_label="'(' + datum['건수'] + ')'")
    chart((rect + pct + cnt).properties(width=alt.Step(60), height=alt.Step(54),
                                        title=alt.TitleParams(title, anchor="start", fontSize=14)), fit=False)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 데이터 불러오기 (모델 재학습 없음)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame | None:
    """팀 실험 기록 results/model_scores.csv"""
    path = RES_DIR / "model_scores.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["순서"] = df.groupby("담당자").cumcount() + 1
    return df


@st.cache_data(show_spinner="Train/Test 데이터를 불러오는 중…")
def load_split():
    """학습 때와 같은 Train/Test 분할. data 폴더를 직접 읽고(01 노트북 구조), 없으면 src/load_data.py 를 씁니다.
    반환: ((X_train, X_test, y_train, y_test), 불러온 방법, 오류 내용)"""
    proc = DATA_DIR / "processed" / "Faults_processed.csv"
    tr_path, te_path = DATA_DIR / "split" / "train_idx.csv", DATA_DIR / "split" / "test_idx.csv"
    try:
        if proc.exists() and tr_path.exists() and te_path.exists():
            df = pd.read_csv(proc)
            target = "target" if "target" in df.columns else df.columns[-1]
            X, y = df.drop(columns=[target]), df[target].astype(str)
            tr, te = pd.read_csv(tr_path).iloc[:, 0].values, pd.read_csv(te_path).iloc[:, 0].values
            return (X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]), "data/processed + data/split", None
        from load_data import load_data
        X_train, X_test, y_train, y_test = load_data()
        to_str = lambda y: (y.iloc[:, 0] if isinstance(y, pd.DataFrame) else pd.Series(y)).astype(str)
        return (X_train, X_test, to_str(y_train), to_str(y_test)), "src/load_data.py", None
    except Exception:
        return None, None, traceback.format_exc()


# ──────────────────────────────────────────────────────────────────────────────
# 5. 모델 불러오기 — 모델을 그대로 저장했든 dict 로 감싸 저장했든 쓸 수 있게 합니다
# ──────────────────────────────────────────────────────────────────────────────
def find_models(obj, path="", depth=0) -> list:
    """저장 파일 안을 따라가며 예측할 수 있는(predict_proba 가 있는) 모델을 모두 찾습니다.
    반환: [(위치, 모델), ...]   예) [("['model']", VotingClassifier)]"""
    if hasattr(obj, "predict_proba"):
        return [(path or "(최상위)", obj)]
    if depth > 4:
        return []
    if isinstance(obj, dict):
        children = obj.items()
    elif isinstance(obj, (list, tuple)):
        children = enumerate(obj)
    else:
        return []
    found = []
    for key, value in children:
        found += find_models(value, f"{path}[{key!r}]", depth + 1)
    return found


def find_class_names(obj):
    """모델이 숫자 라벨(0~6)로 학습된 경우를 위해, 저장 파일에 함께 들어 있는 클래스 이름을 찾습니다."""
    if not isinstance(obj, dict):
        return None
    for key, value in obj.items():
        is_encoder = hasattr(value, "inverse_transform") and hasattr(value, "classes_") and not hasattr(value, "predict_proba")
        if is_encoder:                                                      # LabelEncoder
            return [str(c) for c in value.classes_]
        is_name_list = (isinstance(value, (list, tuple)) and value and all(isinstance(c, str) for c in value)
                        and set(value) <= set(CLASS_ORDER)
                        and any(word in str(key).lower() for word in ("class", "label", "target")))
        if is_name_list:
            return list(value)
        if isinstance(value, dict):
            names = find_class_names(value)
            if names:
                return names
    return None


def pick_models(found: list) -> list:
    """찾은 모델 중 쓸 것을 고릅니다.
    ① Soft Voting 객체가 통째로 있으면 그것 하나  ② 이름에 final/voting/ensemble 이 들어간 모델이 하나면 그것
    ③ 그 밖에는 모두(구성 모델을 따로 저장한 경우 → 확률 평균)"""
    voting = [f for f in found if hasattr(f[1], "voting")]
    if voting:
        return voting[:1]
    named = [f for f in found if any(word in f[0].lower() for word in ("final", "voting", "ensemble"))]
    return named if len(named) == 1 else found


class LoadedModel:
    """저장 파일에서 꺼낸 모델(들)을 하나처럼 쓰는 감싸개. 모델이 여러 개면 예측 확률을 평균합니다."""

    def __init__(self, found: list, class_names: list[str] | None):
        self.paths = [path for path, _ in found]
        self.members = [model for _, model in found]
        self.class_names = class_names
        self.classes_ = self.names_of(self.members[0])

    def names_of(self, model) -> list[str]:
        """모델의 클래스를 이름으로 (숫자 라벨이면 클래스 이름으로 바꿈)."""
        if self.class_names:
            return [self.class_names[int(c)] if isinstance(c, (int, np.integer)) else str(c) for c in model.classes_]
        return [str(c) for c in model.classes_]

    def predict_proba(self, X) -> np.ndarray:
        probs = [pd.DataFrame(m.predict_proba(X), columns=self.names_of(m)).reindex(columns=self.classes_, fill_value=0.0)
                 for m in self.members]
        return (sum(probs) / len(probs)).values

    def expected_columns(self) -> list[str] | None:
        """모델이 학습 때 받은 열 이름 (VotingClassifier 는 하위 모델에서 확인)."""
        for m in self.members:
            for est in [m, *(getattr(m, "estimators_", None) or [])]:
                names = getattr(est, "feature_names_in_", None)
                if names is not None:
                    return [str(c) for c in names]
        return None

    def describe(self) -> str:
        if len(self.members) == 1:
            return f"{type(self.members[0]).__name__} {self.paths[0]}"
        return f"모델 {len(self.members)}개의 확률 평균: " + ", ".join(
            f"{p} ({type(m).__name__})" for p, m in zip(self.paths, self.members))


@st.cache_resource(show_spinner="저장된 최종 모델을 불러오는 중…")
def load_model():
    """models 폴더의 모델 파일을 차례로 열어 처음으로 쓸 수 있는 모델을 돌려줍니다.
    반환: (LoadedModel 또는 None, 사용한 파일·구성 설명, 오류 내용)"""
    import joblib
    errors = []
    for path in MODEL_FILES:
        if not path.exists():
            continue
        try:
            saved = joblib.load(path)
        except Exception:
            errors.append(f"[{path.name}] 불러오기 실패:\n{traceback.format_exc()}")
            continue
        found = find_models(saved)
        if not found:
            content = list(saved.keys()) if isinstance(saved, dict) else type(saved).__name__
            errors.append(f"[{path.name}] 예측 가능한 모델이 없습니다. 저장된 내용: {content}")
            continue
        model = LoadedModel(pick_models(found), find_class_names(saved))
        return model, f"{path.name} → {model.describe()}", None
    if not errors:
        errors.append("모델 파일이 없습니다. 찾아본 위치:\n" + "\n".join(str(p) for p in MODEL_FILES))
    return None, None, "\n\n".join(errors)


# ──────────────────────────────────────────────────────────────────────────────
# 6. 예측 — 입력에 파생변수를 붙이고 모델이 기대하는 열 순서로 맞춥니다
# ──────────────────────────────────────────────────────────────────────────────
PLATE_KEYS = ["Length_of_Conveyer", "Steel_Plate_Thickness", "TypeOfSteel_A300"]   # 생산 조건
PLATE_COLS = ["plate_defect_count", "thickness_per_conveyer"]                    # 강판 맥락 파생변수 2개


def _predict(model: LoadedModel, X: pd.DataFrame) -> pd.DataFrame:
    """모델이 기대하는 열 순서로 맞춰 예측 확률을 계산합니다 (열 = CLASS_ORDER 순서)."""
    cols = model.expected_columns()
    if cols is not None:
        X = X[[c for c in cols if c in X.columns or c not in PLATE_COLS]]   # 강판 맥락 변수는 없으면 빼고 넣음
    proba = pd.DataFrame(model.predict_proba(X), columns=model.classes_, index=X.index)
    order = [c for c in CLASS_ORDER if c in proba.columns] + [c for c in proba.columns if c not in CLASS_ORDER]
    return proba[order]


def predict_proba(model: LoadedModel, X_raw: pd.DataFrame, X_train: pd.DataFrame) -> pd.DataFrame:
    """측정값(원본 26열) → 유형별 예측 확률.
    ① 06 노트북 모델은 강판 맥락 변수 2개를 파이프라인 안에서 직접 계산하므로 파생변수만 붙인 34열을 넣습니다.
    ② 그 방식이 맞지 않으면 2개를 Train 기준으로 계산해(모델 재학습 아님) 36열로 다시 시도합니다."""
    from Taegyeom_model_utils import PlateContextEncoder, add_features
    X34 = add_features(X_raw)
    try:
        return _predict(model, X34)
    except Exception as first:
        try:
            X36 = PlateContextEncoder().fit(add_features(X_train)).transform(X34)
            return _predict(model, X36)
        except Exception as second:
            raise RuntimeError(f"34열 입력 실패: {first!r}\n36열 입력 실패: {second!r}") from second


def show_error(title: str, detail: str | None):
    """실행이 실패하면 원인을 바로 볼 수 있도록 오류 내용과 실행 환경을 보여줍니다."""
    st.error(title)
    with st.expander("오류 내용과 실행 환경 보기", expanded=True):
        if detail:
            st.code(detail[-6000:], language="text")
        versions = {"python": platform.python_version()}
        for mod in ["sklearn", "imblearn", "lightgbm", "xgboost", "joblib", "pandas", "numpy"]:
            try:
                versions[mod] = __import__(mod).__version__
            except Exception:
                versions[mod] = "설치 안 됨"
        st.write("라이브러리 버전 (노트북을 실행한 환경과 같아야 pickle 이 열립니다)")
        st.dataframe(pd.DataFrame([versions]), hide_index=True)
        st.write(f"프로젝트 폴더: `{ROOT}`")
    if st.button("다시 불러오기"):
        load_model.clear()
        load_split.clear()
        st.rerun()
