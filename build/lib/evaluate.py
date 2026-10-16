"""실험 결과 기록 도구. 팀원 전원이 이 함수를 사용한다."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_PATH = BASE_DIR / "results" / "model_scores.csv"

COLUMNS = ["실험명", "담당자", "Macro_F1", "Weighted_F1", "정확도", "비고"]


def log_result(name, y_true, y_pred, author="", note=""):
    """예측 결과를 채점해 results/model_scores.csv 에 누적한다."""
    row = {
        "실험명": name,
        "담당자": author,
        "Macro_F1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "Weighted_F1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "정확도": round(accuracy_score(y_true, y_pred), 4),
        "비고": note,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prev = pd.DataFrame(columns=COLUMNS)
    if RESULT_PATH.exists() and RESULT_PATH.stat().st_size > 0:
        try:
            loaded = pd.read_csv(RESULT_PATH)
            if "실험명" in loaded.columns:
                prev = loaded
            else:
                backup = RESULT_PATH.with_name("model_scores_backup.csv")
                loaded.to_csv(backup, index=False, encoding="utf-8-sig")
                print(f"양식이 달라 백업했습니다 → {backup.name}")
        except Exception as e:
            print("기존 파일 읽기 실패, 새로 시작합니다:", e)

    prev = prev[prev["실험명"] != name]
    out = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    out = out.reindex(columns=COLUMNS)
    out.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    print(pd.DataFrame([row]).to_string(index=False))
    return row


def results_table():
    """누적된 전체 결과를 Macro F1 순으로 반환한다."""
    if not RESULT_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(RESULT_PATH).sort_values("Macro_F1", ascending=False)