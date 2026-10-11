# =====================================================================
#  문제 4 — 같은 일을 도구로, 그리고 직선의 벽을 트리로  (04 · 07)
# =====================================================================
#  [상황]
#    문제 3 에서 손으로 만든 로지스틱 회귀는 F1 0.5 를 못 넘었습니다.
#    특징을 만들어 넣어도 별로 안 올랐죠. 오늘은 그게 '모델의 한계'인지 확인합니다.
#
#  [파일]  데이터/11_설비센서_ai4i_전체.csv
#  [쓰는 것]  sklearn 자유. (04 · 07 에서 본 도구 전부)
#  [공통]  입력 = 센서 다섯 개 (공기온도, 공정온도, 회전수, 토크, 공구마모), 정답 = 고장여부
# =====================================================================

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "데이터")

df = pd.read_csv(os.path.join(DATA, "11_설비센서_ai4i_전체.csv"), encoding="utf-8-sig")

특징 = ["공기온도", "공정온도", "회전수", "토크", "공구마모"]
X = df[특징].values
y = df["고장여부"].values


# ---------------------------------------------------------------------
# [4-1] 문제 3 을 한 줄로
#   문제 3 [3-1] 과 '똑같은' train/test 를 만드세요. (같은 방법, 같은 seed)
#   그다음 '표준화 + 로지스틱 회귀' 를 한 덩어리로 묶어 학습하세요. (max_iter 넉넉히)
#
#   ★ 먼저 예측 ★  가중치가 문제 3 손코드와 완전히 같을까요?
#   확인값: 토크 +2.71, 회전수 +2.04 ... (순서와 부호는 손코드와 같음), 재현율 0.206
#   ★ 설명 ★  크기가 '살짝' 다른 이유 두 가지를 대 보세요. (하나는 걸음 수, 하나는 sklearn 의 기본 옵션)
# ---------------------------------------------------------------------

# ★ 먼저 예측 ★
# 가중치의 부호와 크기 순서는 비슷하지만 완전히 같지는 않을 것으로 예상
# sklearn은 최적화 방식과 기본 규제 조건이 손코드와 다르기 때문

print("[4-1] 문제 3을 sklearn으로")

rng = np.random.RandomState(0)

고장idx = rng.permutation(np.where(y == 1)[0])
정상idx = rng.permutation(np.where(y == 0)[0])

고장수 = int(len(고장idx) * 0.7)
정상수 = int(len(정상idx) * 0.7)

train_idx = np.concatenate([고장idx[:고장수], 정상idx[:정상수]])
test_idx = np.concatenate([고장idx[고장수:], 정상idx[정상수:]])

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

로지스틱 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
로지스틱.fit(X_train, y_train)

print(f"train {len(y_train)} / test {len(y_test)}")

for 이름, w in zip(특징, 로지스틱[-1].coef_[0]):
    print(f"{이름} {w:+.4f}")

print(f"재현율 {recall_score(y_test, 로지스틱.predict(X_test)):.4f}")

# 손코드는 2000번 경사하강했지만 sklearn은 수렴 조건에 따라 반복 횟수가 달라짐
# sklearn의 LogisticRegression은 기본적으로 L2 규제를 사용해 가중치 크기를 제한
# 따라서 부호와 순서는 같아도 가중치 크기는 조금 다를 수 있음


# ---------------------------------------------------------------------
# [4-2] 버그 찾기
#   후배가 아래 코드를 짜 와서 "C=1 이 최고고 test F1 이 0.22 나왔어요" 라고 보고했습니다.
#   (X, y 는 위 공통 입력·정답)
#
#       Z = StandardScaler().fit_transform(X)
#       Z_tr, Z_te, y_tr, y_te = train_test_split(Z, y, test_size=0.3, random_state=0)
#       최고 = None
#       for C in [0.01, 0.1, 1, 10]:
#           m = LogisticRegression(C=C, class_weight="balanced").fit(Z_tr, y_tr)
#           점수 = f1_score(y_te, m.predict(Z_te))
#           if 최고 is None or 점수 > 최고[0]:
#               최고 = (점수, C)
#       print("최고 C:", 최고[1], " 최종 test F1:", 최고[0])
#
#   ★ 설명 ★  잘못된 곳이 세 군데 있습니다. 각각 무엇이 문제고, 그게 '보고된 숫자'를 어떻게 믿을 수 없게 만드는지.
#   그다음 고친 코드를 직접 쓰세요. (stratify, Pipeline, 교차검증으로 C 고르기, test 는 마지막에 한 번)
#   확인: 고친 코드에서 고른 C 와 test F1 을 찍으세요. F1 은 0.24 근처로, 보고(0.22)와 비슷합니다.
#   ★ 설명 ★  숫자가 비슷하게 나왔는데 왜 고쳐야 하나요?
# ---------------------------------------------------------------------

# ① 전체 데이터를 먼저 표준화하면 test의 평균과 표준편차가 학습에 반영되어 데이터 누수 발생
# ② stratify 없이 나누면 고장이 적은 데이터에서 train/test의 고장 비율이 달라질 수 있음
# ③ test F1으로 C를 고르면 test가 모델 선택에 사용되어 최종 평가의 신뢰성이 떨어짐
# 따라서 Pipeline으로 누수를 방지하고 교차검증으로 C를 선택한 뒤 test는 마지막에 한 번만 평가

print("\n[4-2] 버그 수정")

후보 = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
)

# 고장 비율을 유지하는 5겹 교차검증
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

탐색 = GridSearchCV(
    후보, {"logisticregression__C": [0.01, 0.1, 1, 10]}, cv=cv, scoring="f1"
)

# [4-1]에서 고장 비율을 유지하며 나눈 train/test 재사용
탐색.fit(X_train, y_train)

print(f"최적 C {탐색.best_params_['logisticregression__C']}")
print(f"교차검증 F1 {탐색.best_score_:.4f}")
print(f"최종 test F1 {f1_score(y_test, 탐색.predict(X_test)):.4f}")

# 결과가 비슷하더라도 잘못된 방법은 test 정보가 학습과 모델 선택에 영향을 줄 수 있음
# 이번에 숫자가 비슷한 것은 우연일 수 있으므로 올바른 검증 절차를 지켜야 함


# ---------------------------------------------------------------------
# [4-3] 결정트리 — 깊이를 test 없이 고르기
#   [4-1] 의 train 만 써서, 깊이 2 / 3 / 4 / 6 / 8 / 12 / 제한없음 각각의
#     - train F1
#     - train 안 5겹 교차검증 F1 평균
#   을 표로 찍으세요. (random_state=0)
#
#   ★ 먼저 예측 ★  깊이 '제한없음' 의 train F1 은?
#   확인값: 제한없음 train F1 1.000 / 교차검증은 깊이 8 에서 최고 (≈ 0.655)
#   교차검증으로 고른 깊이로 train 전체를 다시 학습하고, 그때 처음으로 test F1 을 찍으세요.
#   확인값: test F1 ≈ 0.70
#
#   ★ 설명 ★  train F1 로 깊이를 골랐다면 몇을 골랐을까요? 그게 왜 나쁜가요?
# ---------------------------------------------------------------------

# ★ 먼저 예측 ★
# 깊이 제한이 없으면 학습 데이터를 끝까지 분할해 외울 수 있으므로 train F1은 1.0에 가까울 것

print("\n[4-3] 결정트리 깊이 비교")
print(f"{'깊이':<12} {'train F1':>12} {'CV F1':>12}")

최고점수 = -1
최적깊이 = None

for 깊이 in [2, 3, 4, 6, 8, 12, None]:
    트리 = DecisionTreeClassifier(max_depth=깊이, random_state=0)
    트리.fit(X_train, y_train)

    train_f1 = f1_score(y_train, 트리.predict(X_train))
    cv_f1 = cross_val_score(트리, X_train, y_train, cv=5, scoring="f1").mean()

    print(f"{str(깊이):<12} {train_f1:>12.4f} {cv_f1:>12.4f}")

    if cv_f1 > 최고점수:
        최고점수 = cv_f1
        최적깊이 = 깊이

트리 = DecisionTreeClassifier(max_depth=최적깊이, random_state=0)
트리.fit(X_train, y_train)

print(f"최적 깊이 {최적깊이}")
print(f"최고 교차검증 F1 {최고점수:.4f}")
print(f"최종 test F1 {f1_score(y_test, 트리.predict(X_test)):.4f}")

# train F1만 보면 깊이 제한없음을 선택하게 됨
# 하지만 학습 데이터를 지나치게 세밀하게 분할하면 잡음까지 외우는 과적합 발생
# 따라서 train 점수가 아니라 교차검증 점수로 깊이를 선택해야 함


# ---------------------------------------------------------------------
# [4-4] 네 모델 붙여 보기
#   같은 train/test 로 아래 넷의 재현율·정밀도·F1 을 표로 찍으세요.
#     로지스틱(표준화 + class_weight="balanced"), 결정트리(4-3 에서 고른 깊이),
#     랜덤포레스트(n_estimators=300), 부스팅(HistGradientBoosting)   — 전부 random_state=0
#   확인값: 로지스틱 F1 ≈ 0.25 (재현율은 0.83 으로 높지만 정밀도 0.14) / 트리 계열 셋은 전부 0.70 근처
#
#   ★ 설명 ★  07 에서는 '공정온도 예측' 에서 선형이 이겼고, 오늘은 트리가 크게 이겼습니다.
#             이 데이터에서 트리가 이긴 이유를 [3-4] 의 힌트 두 개와 연결해서 설명하세요.
# ---------------------------------------------------------------------
print("\n[4-4] 네 모델 비교")

모델들 = [
    (
        "로지스틱",
        make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
        ),
    ),
    ("결정트리", DecisionTreeClassifier(max_depth=최적깊이, random_state=0)),
    ("랜덤포레스트", RandomForestClassifier(n_estimators=300, random_state=0)),
    ("부스팅", HistGradientBoostingClassifier(random_state=0)),
]

print(f"{'모델':<14} {'재현율':>10} {'정밀도':>10} {'F1':>10}")

for 이름, 모델 in 모델들:
    모델.fit(X_train, y_train)
    판정 = 모델.predict(X_test)

    재현율 = recall_score(y_test, 판정)
    정밀도 = precision_score(y_test, 판정, zero_division=0)
    F1 = f1_score(y_test, 판정, zero_division=0)

    print(f"{이름:<14} {재현율:>10.4f} {정밀도:>10.4f} {F1:>10.4f}")

# 공정온도는 센서 간 관계가 거의 직선이므로 선형회귀가 유리했음
# 하지만 고장은 전력이 너무 낮거나 높을 때 모두 발생하는 비선형 관계를 가짐
# 또한 온도차가 작고 회전수가 낮은 경우처럼 여러 조건이 동시에 작용
# 트리는 값의 범위를 나누고 조건을 조합할 수 있어 이러한 고장 패턴에 유리함


# ---------------------------------------------------------------------
# [4-5] 트리가 찾은 규칙 읽기
#   깊이 3 짜리 트리를 train 으로 학습해서 규칙을 글로 뽑으세요. (export_text)
#   그리고 랜덤포레스트의 특징 중요도를 크기순으로 찍으세요.
#
#   ★ 설명 ★  규칙에서 '토크' 가 어떤 식으로 쓰였나요? (한 번만? 위아래 양쪽으로?)
#             로지스틱 회귀는 토크에 가중치 '+2.7' 하나만 줄 수 있었습니다. 이게 왜 한계인가요?
#   ★ 설명 ★  규칙에 회전수 1380 근처의 갈림이 나옵니다. 이건 어느 고장 종류와 관련 있을까요?
#             (고장 종류별로 회전수 평균을 직접 확인해 보세요)
# ---------------------------------------------------------------------
print("\n[4-5] 트리가 찾은 규칙")

규칙트리 = DecisionTreeClassifier(max_depth=3, random_state=0)
규칙트리.fit(X_train, y_train)

print(export_text(규칙트리, feature_names=특징))

숲 = RandomForestClassifier(n_estimators=300, random_state=0)
숲.fit(X_train, y_train)

중요도 = pd.Series(숲.feature_importances_, index=특징)
print(f"랜덤포레스트 특징 중요도\n{중요도.sort_values(ascending=False).round(4)}")

# 고장 종류별 회전수 평균 확인
고장종류 = ["공구마모고장", "방열고장", "전력고장", "과부하고장", "무작위고장"]

print("\n고장 종류별 회전수 평균")
for 종류 in 고장종류:
    평균 = df.loc[df[종류] == 1, "회전수"].mean()
    print(f"{종류} {평균:.2f}")

print(f"전체 회전수 평균 {df['회전수'].mean():.2f}")

# 토크가 13.15 이하로 너무 낮거나 60.55 초과로 너무 높을 때 고장 위험이 나타남
# 높은 토크 구간에서는 공구마모와 토크 66.60 기준으로 다시 분기
# 로지스틱 회귀는 토크에 양수 가중치 하나만 주므로 높을수록 위험하다는 한 방향만 표현
# 따라서 토크가 너무 낮아도 위험한 관계를 표현하기 어려움

# 회전수 1380.5 부근의 분기는 전력고장과 관련 있음
# 전력고장은 토크와 회전수로 계산되는 전력이 너무 낮거나 높을 때 발생
# 따라서 낮은 회전수와 높은 토크의 조합이 전력고장 조건에 영향을 줄 수 있음
