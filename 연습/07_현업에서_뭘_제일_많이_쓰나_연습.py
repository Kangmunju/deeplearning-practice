import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "수업용_데이터")

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import recall_score, precision_score, f1_score


df = pd.read_csv(os.path.join(DATA, "11_설비센서_ai4i.csv"), encoding="utf-8-sig")
feature_name = ["공기온도", "회전수", "토크", "공구마모"]
X = df[feature_name].values
y_temp = df["공정온도"].values
y_broken = df["고장여부"].values


# =====================================================================
# 0. 오늘 새로 꺼내는 도구들 — 미리 한눈에
# =====================================================================
# 04 §0 의 표에 트리 서랍 두 개가 더해집니다. 나머지(train_test_split, StandardScaler,
# make_pipeline, 채점 함수들)는 04 에서 쓰던 그대로예요.
#
#   sklearn.tree       트리 한 그루
#   sklearn.ensemble   트리 여러 그루를 합친 것
#
#  [트리 한 그루]
#   DecisionTreeClassifier(max_depth=2)   종류를 맞히는 트리 (분류)
#   DecisionTreeRegressor(max_depth=3)    숫자를 맞히는 트리 (회귀)
#     max_depth = 질문을 몇 단계까지 할까. ★ 이걸 안 걸면 통째로 외웁니다 (§2) ★
#     class_weight="balanced" = 드문 쪽을 무겁게 세라 (03 에서 배운 불균형 처방)
#   export_text(트리, feature_names=...)  학습된 트리를 사람이 읽는 글로 뽑아 줍니다 (§1)
#
#  [트리 여러 그루 = 앙상블]
#   RandomForestClassifier(n_estimators=200)   트리를 200그루 만들어 투표시킵니다 (분류)
#   RandomForestRegressor(n_estimators=200)    같은 방식으로 숫자 맞히기 (회귀)
#     n_estimators = 몇 그루 만들까. 많을수록 안정적이고 느립니다. 보통 100~500
#   HistGradientBoostingClassifier / Regressor  부스팅. 틀린 것만 다음 트리가 고쳐 나갑니다
#     XGBoost, LightGBM 과 같은 방식인데 사이킷런에 들어 있어 따로 설치가 필요 없습니다
#
#  [트리만 주는 것]
#   모델.feature_importances_   어느 특징이 답을 가르는 데 많이 쓰였나 (§4-1)
#     02 에서 가중치 크기로 읽던 것과 같은 목적인데, 트리는 이렇게 바로 줍니다
#
# 트리 쪽 규칙 두 가지
#   1) 표준화가 필요 없습니다. "얼마보다 큰가" 만 보니까요 (§1-1 에서 확인)
#   2) 만들기 → fit → predict 는 04 의 모델들과 똑같습니다. 바꿔 끼우기만 하면 됩니다


# =====================================================================
# 1. 결정트리 — 스무고개를 코드로 옮긴 것
# =====================================================================
print("1. 결정트리")

# random_state=0으로 결정트리를 만드는 과정에서 필요한 랜덤 선택을 0번 패턴으로 고정
tree = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=0)
tree.fit(X, y_broken)
print("[1] 학습된 트리를 글로 뽑기 (class 1 = 고장)")
print(f"{export_text(tree, feature_names=feature_name)}")

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    X, y_broken, test_size=0.3, random_state=3, stratify=y_broken
)
# random_state=0으로 랜덤포레스트가 여러 나무를 만드는 과정의 랜덤 선택을 0번 패턴으로 고정
rf_raw = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xc_tr, yc_tr)
rf_std = make_pipeline(
    StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=0)
).fit(Xc_tr, yc_tr)
print(f"[1-1] 트리는 표준화가 불요")
print(f"표준화를 하지 않은 경우의 정확도 {rf_raw.score(Xc_te, yc_te):.4f}")
print(f"표준화를 한 경우의 정확도 {rf_std.score(Xc_te, yc_te):.4f} <- 동일함")

# 1. 결정트리
# [1] 학습된 트리를 글로 뽑기 (class 1 = 고장)
# |--- 공구마모 <= 211.50
# |   |--- 회전수 <= 1740.00
# |   |   |--- class: 0
# |   |--- 회전수 >  1740.00
# |   |   |--- class: 1
# |--- 공구마모 >  211.50
# |   |--- 공기온도 <= 301.40
# |   |   |--- class: 1
# |   |--- 공기온도 >  301.40
# |   |   |--- class: 0
# [1-1] 트리는 표준화가 불요
# 표준화를 하지 않은 경우의 정확도 0.9667
# 표준화를 한 경우의 정확도 0.9667 <- 동일함


# =====================================================================
# 2. 트리 하나는 위험하다 — 깊이를 안 막으면 통째로 외운다
# =====================================================================
print()
print("\n2. 트리 하나는 위험하다")

# random_state=42로 train/test 데이터를 랜덤으로 나누는 것을 42번 패턴으로 고정
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y_temp, test_size=0.3, random_state=42)
print("[2] 트리 깊이를 바꿔가며 (공정온도 예측)")
for depth in [2, 3, 5, None]:
    t = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(Xr_tr, yr_tr)
    name = "제한 없음" if depth is None else f"깊이 {depth}"
    print(
        f"{name} -> train {round(t.score(Xr_tr, yr_tr), 4)} / test {round(t.score(Xr_te, yr_te), 4)}"
    )

# 2. 트리 하나는 위험하다
# [2] 트리 깊이를 바꿔가며 (공정온도 예측)
# 깊이 2 -> train 0.7584 / test 0.6028
# 깊이 3 -> train 0.8348 / test 0.6649
# 깊이 5 -> train 0.9241 / test 0.4187
# 제한 없음 -> train 1.0 / test 0.3418


# =====================================================================
# 3. 트리 하나가 위험하면 여러 개를 만들어 투표시킨다
# =====================================================================
print()
print("\n3. 트리 하나가 위험하면 여러 개를 만들어 투표시킨다")

print("[3] 한 그루 vs 숲 vs 부스팅 (공정온도 예측)")
for name, model in [
    ("트리 한 그루(제한 없음)", DecisionTreeRegressor(random_state=0)),
    ("랜덤포레스트 200그루", RandomForestRegressor(n_estimators=200, random_state=0)),
    ("부스팅", HistGradientBoostingRegressor(random_state=0)),
]:
    m = model.fit(Xr_tr, yr_tr)
    print(
        f"{name} train {round(m.score(Xr_tr, yr_tr), 4)} / test {round(m.score(Xr_te, yr_te), 4)}"
    )

# 3. 트리 하나가 위험하면 여러 개를 만들어 투표시킨다
# [3] 한 그루 vs 숲 vs 부스팅 (공정온도 예측)
# 트리 한 그루(제한 없음) train 1.0 / test 0.3418
# 랜덤포레스트 200그루 train 0.9659 / test 0.6434
# 부스팅 train 0.9061 / test 0.6101


# =====================================================================
# 4. 그래서 뭐가 더 좋은가 — 두 판 붙여 본다
# =====================================================================
print()
print("\n4. 그래서 뭐가 더 좋은가")

# ── 1판. ai4i 공정온도 예측 (200행, 센서 4개) ──
print("[4]")
print("-" * 64)
print("1판 - 공정온도 예측 (200행, 관계가 거의 직선)")
for name, model in [
    ("선형회귀", make_pipeline(StandardScaler(), LinearRegression())),
    ("랜덤포레스트", RandomForestRegressor(n_estimators=200, random_state=0)),
    ("부스팅", HistGradientBoostingRegressor(random_state=0)),
]:
    m = model.fit(Xr_tr, yr_tr)
    print(
        f"{name} train {round(m.score(Xr_tr, yr_tr), 4)} / test {round(m.score(Xr_te, yr_te), 4)}"
    )
print(
    "선형회귀 승 : 00에서 본 상관 0.89 -> 관계가 직선이기 때문에 직선 모델이 제일 잘 맞음"
)
print("데이터가 200행뿐이라 트리가 학습할 데이터가 부족함")

# 4. 그래서 뭐가 더 좋은가
# [4]
# ----------------------------------------------------------------
# 1판 - 공정온도 예측 (200행, 관계가 거의 직선)
# 선형회귀 train 0.8155 / test 0.7425
# 랜덤포레스트 train 0.9659 / test 0.6434
# 부스팅 train 0.9061 / test 0.6101
# 선형회귀 승 : 00에서 본 상관 0.89 -> 관계가 직선이기 때문에 직선 모델이 제일 잘 맞음
# 데이터가 200행뿐이라 트리가 학습할 데이터가 부족함

# ── 2판. 압연 고진동 분류 (570행, 특징 10개) ──
r = pd.read_csv(os.path.join(DATA, "T-CR1-SPM01_압연특징.csv"), encoding="utf-8-sig")
threshold = r["VIB-BOT_RMS"].quantile(0.90)  # 경계 상위 10%
y_고진동 = (r["VIB-BOT_RMS"] >= threshold).astype(int).values
입력열 = [c for c in r.columns if c.startswith(("VIB-TOP", "CUR-MTR"))]
Xv = r[입력열].values
Xv_tr, Xv_te, vy_tr, vy_te = train_test_split(
    Xv, y_고진동, test_size=0.3, random_state=42, stratify=y_고진동
)

print("-" * 64)
print(f"2판 - 압연 고진동 분류 (570행, 특징 {len(입력열)}개)")
for name, model in [
    (
        "로지스틱 회귀",
        make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
    ),
    (
        "랜덤포레스트",
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=0
        ),
    ),
    ("부스팅", HistGradientBoostingClassifier(random_state=0)),
]:
    m = model.fit(Xv_tr, vy_tr)
    p = m.predict(Xv_te)
    print(
        f"{name} 재현율 {round(recall_score(vy_te, p), 2)} 정밀도 {round(precision_score(vy_te, p, zero_division=0), 2)} F1 {round(f1_score(vy_te, p, zero_division=0), 2)}"
    )
print(
    "랜덤포레스트 승 : F1 0.80 VS 0.62 -> 특징이 10개고 관계가 직선이 아니기 때문에 트리가 유리"
)

# ----------------------------------------------------------------
# 2판 - 압연 고진동 분류 (570행, 특징 10개)
# 로지스틱 회귀 재현율 0.88 정밀도 0.48 F1 0.62
# 랜덤포레스트 재현율 0.82 정밀도 0.78 F1 0.8
# 부스팅 재현율 0.53 정밀도 0.56 F1 0.55
# 랜덤포레스트 승 : F1 0.80 VS 0.62 -> 특징이 10개고 관계가 직선이 아니기 때문에 트리가 유리


forest = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=0
).fit(Xv, y_고진동)
print("[4-1] 랜덤포레스트가 꼽은 중요한 특징 5개")
for name, val in sorted(zip(입력열, forest.feature_importances_), key=lambda t: -t[1])[
    :5
]:
    print(f"{name} {round(val, 3)}")

# [4-1] 랜덤포레스트가 꼽은 중요한 특징 5개
# VIB-TOP_STD 0.231
# VIB-TOP_RMS 0.205
# VIB-TOP_PTP 0.174
# CUR-MTR_PTP 0.093
# CUR-MTR_RMS 0.091


# =====================================================================
# 5. 현업에서 실제로 하는 일 — 모델 고르는 절차
# =====================================================================
print()
print("\n5. 현업에서 실제로 하는 일")

candidates = [
    ("선형회귀", make_pipeline(StandardScaler(), LinearRegression())),
    ("랜덤포레스트", RandomForestRegressor(n_estimators=200, random_state=0)),
    ("부스팅", HistGradientBoostingRegressor(random_state=0)),
]
print("[5] 후보를 교차검증으로 한꺼번에 (공정온도 예측, test는 손도 안 댐)")
for name, model in candidates:
    cv = cross_val_predict(model, Xr_tr, yr_tr, cv=5)
    print(f"{name}   CV 평균 {round(cv.mean(), 3)}  (조각별 {cv.round(2)})")

# 5. 현업에서 실제로 하는 일
# [5] 후보를 교차검증으로 한꺼번에 (공정온도 예측, test는 손도 안 댐)
# 선형회귀   CV 평균 310.154  (조각별 [314.04 310.96 311.43 309.72 309.07 311.95 309.27 308.68 312.52 311.61
#  313.15 308.78 310.1  309.33 310.28 309.1  307.61 308.41 311.74 307.81
#  309.91 311.14 308.69 309.62 311.15 307.68 308.71 306.17 309.89 310.96
#  309.75 312.38 310.35 306.72 312.93 308.64 310.68 308.33 314.76 312.47
#  310.23 314.03 313.23 311.57 312.09 310.64 309.72 307.52 310.45 308.6
#  312.5  312.05 308.07 311.05 312.56 308.47 312.26 312.17 308.92 311.79
#  314.61 307.66 311.05 309.96 308.46 314.75 308.94 309.75 308.18 311.53
#  311.2  312.19 309.51 310.55 310.32 307.54 308.92 308.27 308.56 312.17
#  310.6  313.66 308.55 306.09 311.92 309.5  309.62 309.16 310.37 309.67
#  311.69 308.27 309.24 308.77 310.52 310.3  316.01 308.88 313.17 308.14
#  310.74 309.8  311.87 309.7  310.65 309.51 306.42 307.38 307.98 308.03
#  310.63 308.79 313.1  307.47 311.6  311.04 311.48 312.14 310.69 311.39
#  309.73 311.08 307.54 308.   310.38 311.88 308.39 307.94 308.82 310.6
#  308.68 307.59 311.31 307.1  309.64 310.03 307.37 314.99 310.45 306.92])
# 랜덤포레스트   CV 평균 310.136  (조각별 [313.51 311.45 311.32 310.25 309.94 312.62 309.95 308.23 312.37 311.25
#  312.57 308.58 310.72 310.18 310.57 309.84 307.18 307.81 312.38 307.86
#  310.79 310.86 308.65 310.34 311.87 307.58 308.74 306.73 310.04 310.66
#  310.07 312.63 311.05 306.23 312.49 308.41 311.09 307.99 314.3  312.59
#  310.23 313.39 312.45 311.05 312.45 310.66 309.93 307.26 310.61 308.32
#  312.31 312.55 308.15 310.26 312.59 307.14 312.76 312.79 309.55 312.22
#  314.47 306.99 310.9  310.26 308.2  314.38 309.42 309.6  308.18 311.85
#  310.52 312.53 309.66 310.82 310.27 307.66 309.42 308.28 308.3  312.93
#  310.43 313.11 308.65 306.59 312.53 309.19 309.11 309.64 310.65 308.9
#  312.45 306.85 309.07 308.71 310.81 309.79 313.87 309.68 312.79 308.08
#  310.64 309.98 312.22 309.25 310.21 309.16 307.12 307.73 308.09 307.61
#  310.   308.6  312.66 306.7  311.7  311.06 310.44 312.36 311.2  311.88
#  309.89 310.55 307.34 307.75 310.26 312.19 308.2  307.97 309.27 310.32
#  308.21 307.39 311.8  307.03 309.5  310.73 306.89 313.97 310.49 306.67])
# 부스팅   CV 평균 310.235  (조각별 [312.77 311.66 311.12 310.11 309.48 311.74 310.92 307.95 313.26 311.06
#  313.79 308.33 310.87 310.48 310.89 309.17 307.43 308.05 311.84 307.98
#  310.88 310.75 308.94 310.44 311.68 307.85 308.19 307.35 310.17 310.49
#  309.5  312.72 310.62 307.56 312.98 307.93 311.04 307.69 313.41 312.64
#  310.17 313.21 312.87 311.4  312.74 310.86 309.77 307.81 310.52 308.52
#  313.4  312.53 308.07 310.07 313.4  308.38 313.71 313.58 309.93 312.49
#  312.99 306.85 310.96 310.26 308.9  313.11 309.97 309.47 307.17 311.92
#  311.19 313.41 310.01 310.63 310.32 307.21 309.86 308.17 308.6  313.08
#  310.62 313.5  308.91 307.57 313.2  309.37 309.33 309.54 310.3  309.79
#  311.62 307.01 309.74 309.02 310.82 310.44 313.13 309.41 313.44 307.17
#  310.84 310.06 312.62 310.   309.65 309.09 307.24 307.57 308.04 307.8
#  310.24 308.55 312.7  307.98 312.2  311.34 311.42 313.24 311.57 311.2
#  309.18 310.88 308.13 307.65 310.1  312.16 307.51 307.19 309.44 309.56
#  308.56 307.41 312.15 308.57 309.04 310.57 307.66 312.57 309.99 308.11])


# =====================================================================
# 6. 정리 — 언제 뭘 쓰나, 그리고 제일 많이 치는 메서드
# =====================================================================
print()
print()
print("""
[6] 모델 고르는 기준

    상황                                   먼저 써 볼 것
    ------------------------------------   ---------------------------------
    행이 몇백 개, 관계가 단순해 보임        선형회귀 / 로지스틱 회귀
    설명이 제일 중요 (현장 보고, 규제 대응)  결정트리 (얕게) / 선형모델
    표 데이터, 행 수천~수백만, 특징 많음     랜덤포레스트 -> 부스팅
    캐글 같은 데서 점수를 끝까지 짜야 함     부스팅 (XGBoost / LightGBM)
    이미지, 음성, 자연어, 시계열 파형        딥러닝 (파이토치, 05)

    한 줄 요약: 표 데이터면 사이킷런 안에서 끝납니다.
              선형모델로 기준선을 잡고, 랜덤포레스트로 올려 보고, 필요하면 부스팅.

    ※ 제일 흔한 실수는 '좋은 모델을 고르는 것'에 시간을 쏟는 겁니다.
      실제로 점수를 올리는 건 대부분 (1) 데이터를 더 모으는 것 (2) 특징을 잘 만드는 것
      (3) 라벨을 제대로 정의하는 것 입니다. 모델 교체는 보통 마지막에 조금 올려 줍니다.
      06 에서 '고장 3일 전'이 안 됐던 거 기억하시죠? 그건 모델을 바꿔도 안 됩니다.

[6-1] 현업에서 제일 많이 치는 것들 (외울 순서대로)

    pandas -- 실무 시간의 절반 이상이 여기서 갑니다
      pd.read_csv(경로, encoding="utf-8-sig")   파일 열기. 한글이면 encoding 필수
      df.shape / df.head() / df.dtypes          열자마자 이 셋. 습관으로 만드세요
      df.describe()                             min/max 로 이상값을 제일 빨리 찾음
      df["열"].value_counts()                   문자열 열은 이걸로
      df[df["열"] > 값]                          조건으로 행 고르기
      df.isna().sum()                           결측 세기
      df.drop_duplicates()                      중복 제거. 안 하면 04 의 누수 사고
      df.groupby("열").mean()                   그룹별 비교. 06 의 위험 vs 평소가 이것
      pd.to_datetime(df["열"])                  시각 열은 무조건 변환부터
      df.merge(other, on="키")                  표 두 개 붙이기. 06 의 정비이력 결합

    scikit-learn -- 모델 쪽은 사실 몇 개 안 됩니다
      train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
      make_pipeline(StandardScaler(), 모델)     누수를 구조적으로 막는 기본형
      .fit(X, y) / .predict(X) / .predict_proba(X)[:, 1]
      cross_val_score(모델, X, y, cv=5)         점수 하나를 믿지 않기
      classification_report(y, 예측)            불균형이면 정확도 대신 이것
      RandomForestClassifier / Regressor        표 데이터 기본 후보
      GridSearchCV                              튜닝은 맨 마지막
      joblib.dump / load                        배포

    이 목록이 실무의 90% 입니다. 오늘 전부 한 번씩 쳐 보셨습니다.
""")

#  현업 표 데이터의 기본 후보는 랜덤포레스트와 부스팅이다. 표준화가 필요 없고, 규칙을 설명할 수 있고,
#  특징 중요도를 준다. 단, 관계가 직선이거나 데이터가 적으면 선형모델이 이긴다 (오늘 1판이 그랬다).
#  그래서 현업이 하는 일은 '좋은 모델을 아는 것'이 아니라 '후보를 돌려 보고 고르는 것'이다.
#  그리고 점수를 실제로 올리는 건 모델 교체가 아니라 데이터 · 특징 · 라벨 정의다.


# =====================================================================
# 실습 — 트리 깊이와 성능
# =====================================================================
# [문제] 2번에서 깊이 2, 3, 5, 제한없음을 봤습니다.
#        깊이 1 은 어떨까요? 깊이 1, 2, 3 의 train / test 를 나란히 출력하세요.
#
#   힌트: DecisionTreeRegressor(max_depth=d, random_state=0) 로 for 문을 돌리면 됩니다.
print()
print("\n실습 — 트리 깊이와 성능")
for depth in [1, 2, 3]:
    t = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(Xr_tr, yr_tr)
    print(
        f"깊이 {depth} -> train {round(t.score(Xr_tr, yr_tr), 4)} / test {round(t.score(Xr_te, yr_te), 4)}"
    )

#
# [정답]
#   for d in [1, 2, 3]:
#       t = DecisionTreeRegressor(max_depth=d, random_state=0).fit(Xr_tr, yr_tr)
#       print(f"깊이 {d} -> train {t.score(Xr_tr, yr_tr):.3f} test {t.score(Xr_te, yr_te):.3f}")
#
# [나오는 답]
#   깊이 1 -> train 0.550  test 0.330
#   깊이 2 -> train 0.758  test 0.603
#   깊이 3 -> train 0.835  test 0.665
#
# [읽는 법]
#   깊이 1 은 질문을 딱 한 번만 하는 트리입니다. train 0.55, test 0.33. 둘 다 낮죠.
#   이게 뭐였죠? -> 과소적합. 02 에서 배운 그것입니다.
#   2번 표와 이어서 전체 그림을 그려 보면:
#     깊이 1 : 0.550 / 0.330   과소적합 (너무 단순)
#     깊이 3 : 0.835 / 0.665   제일 나음
#     깊이 5 : 0.924 / 0.419   과적합 시작
#     제한없음 : 1.000 / 0.342  완전 암기
#     "왼쪽 끝도 나쁘고 오른쪽 끝도 나쁩니다. 가운데 어딘가가 제일 좋아요."
#     "그 '어딘가' 를 어떻게 찾느냐? 04 에서 배운 GridSearchCV 로 찾습니다.
#      max_depth 를 후보로 주고 교차검증으로 고르면 됩니다."
