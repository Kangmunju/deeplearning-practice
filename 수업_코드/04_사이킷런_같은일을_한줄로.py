# =====================================================================
#  04. 사이킷런 — 01~03 에서 손으로 한 일을 '한 줄'로, 그리고 언제 쓰나
#  실행: python 04_사이킷런_같은일을_한줄로.py   (수업코드 폴더에서)
# =====================================================================
# 사이킷런(scikit-learn, 코드에서는 sklearn) : 파이썬 머신러닝 도구 상자
# 우리가 01~03에서 손으로 짠 것(표준화, 경사하강, 회귀, 분류, 채점)이 전부 함수로 들어있음

# 직접 구현해 본 이유
# 도구가 '안에서 무슨 일을 하는지' 알아야 결과를 읽고 이상이 있을 때 고칠 수 있기 때문

# 표(엑셀 등) 데이터 + 선형 모델 트리 같은 전통 머신러닝에서 사용

# '손으로 한 값 == 사이킷런 값'을 눈으로 확인하고 규칙 3개를 익힐 것

# 규칙
# 1) X는 (설비 수, 센서 수) 표 모양, y는 한 줄
# 2) 만들기 -> fit(학습) -> predict(예측)
# 3) 학습 결과는 이름 끝에 밑줄 : coef_, intercept_

import os
import numpy as np
import pandas as pd

DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "수업용_데이터"
)  # data/수업용데이터/
df = pd.read_csv(os.path.join(DATA, "11_설비센서_ai4i.csv"), encoding="utf-8-sig")
특징이름 = ["공기온도", "회전수", "토크", "공구마모"]


# =====================================================================
# 0. 사이킷런에서 꺼내 쓰는 것들 — 미리 한눈에
# =====================================================================
# ★ 문법 ★ from sklearn.어느상자 import 도구
#           사이킷런은 큰 도구함이라 통째로 부르지 않고, 필요한 서랍에서 필요한 것만 꺼냅니다.
#           서랍 이름(sklearn 뒤에 오는 것)만 알면 어디서 뭘 꺼낼지 감이 옵니다.
#
#   서랍 이름                뜻              여기서 꺼내는 것
#   ---------------------   -------------   --------------------------------------------
#   sklearn.linear_model    직선 계열 모델   LinearRegression, LogisticRegression, Ridge
#   sklearn.tree            트리 하나        DecisionTreeClassifier/Regressor, export_text   (07)
#   sklearn.ensemble        트리 여러 개     RandomForest..., HistGradientBoosting...        (07)
#   sklearn.neighbors       가까운 이웃      KNeighborsRegressor                             (§3-1)
#   sklearn.preprocessing   전처리          StandardScaler, PolynomialFeatures
#   sklearn.model_selection 나누기/고르기    train_test_split, cross_val_score, GridSearchCV
#   sklearn.metrics         채점            confusion_matrix, classification_report, recall/precision/f1_score
#   sklearn.pipeline        이어 붙이기      make_pipeline
#
# 하나씩 무슨 일을 하는지 (여기서 다 보고 가면 뒤가 편합니다)
#
#  [모델] — 만들고(생성) → fit(학습) → predict(예측) 세 동사는 전부 똑같습니다
#   LinearRegression()        숫자를 맞히는 직선 모델. 01·02 에서 손으로 짠 그것 (회귀)
#   LogisticRegression()      종류를 맞히는 모델. 03 의 sigmoid + 로그손실이 이 안에 (분류)
#                             기본으로 규제가 켜져 있고, max_iter 는 "몇 걸음까지 걸을까"
#   Ridge(alpha=1.0)          가중치가 커지면 벌점을 주는 선형회귀. alpha 가 벌점 세기 (과적합 처방)
#   KNeighborsRegressor(1)    "제일 비슷한 것 하나 찾아 그 답을 그대로 말하기". §3-1 누수 시연용
#
#  [전처리] — fit 으로 기준을 외우고, transform 으로 값을 바꿉니다
#   StandardScaler()          02 의 표준화. fit 하면 학습용 평균·표준편차를 외우고,
#                             transform 하면 그 자로 값을 바꿉니다. ★ fit 은 학습용에만 ★
#   PolynomialFeatures(4)     특징끼리 곱하고 제곱해 개수를 불립니다. §6-1 에서 과적합 만들 때만 사용
#
#  [나누기 · 고르기]
#   train_test_split(...)     02 의 "섞고 70:30 자르기" 한 줄. random_state 로 매번 같게,
#                             stratify=y 로 고장 비율을 양쪽에 똑같이 (03 에서 손으로 한 일)
#   cross_val_score(...)      학습용을 cv 조각으로 나눠 번갈아 채점하고 점수 목록을 줍니다.
#                             test 는 건드리지 않습니다 (§7)
#   GridSearchCV(...)         후보값을 전부 교차검증으로 돌려 제일 좋은 걸 고르고 다시 학습까지 (§7)
#
#  [채점]
#   confusion_matrix(y, 예측)      네 칸 표. 사이킷런 순서는 [[TN, FP], [FN, TP]] 로 03 과 자리가 다릅니다
#   classification_report(...)     정밀도·재현율·f1 을 한 표로. 불균형이면 정확도 대신 이걸 봅니다
#   recall_score(y, 예측)          재현율 = 실제 고장 중 몇 개를 잡았나 (놓치면 안 될 때)
#   precision_score(y, 예측)       정밀도 = 고장이라 외친 것 중 몇 개가 진짜였나 (헛경보가 곤란할 때)
#   f1_score(y, 예측)              재현율과 정밀도를 하나로 합친 점수 (07 에서 모델 비교할 때)
#
#  [이어 붙이기]
#   make_pipeline(A, B)       A 를 거쳐 B 로 가는 한 덩어리 모델. fit 하면 A 는 학습용으로만
#                             기준을 잡고, predict 하면 변환만 합니다 → 누수를 구조적으로 차단 (§3)
#
# 공통 규칙 세 가지 (이것만 외우면 나머지는 문서 보고 씁니다)
#   1) X 는 (행 수, 열 수) 2차원 표, y 는 1차원 한 줄
#   2) 만들기 → .fit(X, y) → .predict(X). 전처리 도구는 .fit → .transform
#   3) 학습으로 알아낸 값은 이름 끝에 밑줄: .coef_, .intercept_, .mean_, .feature_importances_


# =====================================================================
# 1. 회귀 — 01 의 "공기온도 → 공정온도" 를 세 줄로
# =====================================================================
from sklearn.linear_model import LinearRegression

x = df["공기온도"].values
y = df["공정온도"].values
X1 = x.reshape(-1, 1)
# 규칙 1) 센서가 하나여도 (200, 1) '세로 표'로 세움
# reshape(-1, 1)의 -1 : 행 수를 알아서 맞출 것

print("[1] x.shape", x.shape, "→ X1.shape", X1.shape)
# 사이킷런은 "행 = 설비, 열 = 센서" 표만 받음
# 센서 1개면 열이 1개인 표 -> 한 줄 짜리 배열을 받지 않음!

# 규칙 2) 만들고 학습
model = LinearRegression()  # ① 만들기 — 빈 선형회귀 모델. 아직 아무것도 모름
model.fit(X1, y)  # ② 학습 — 01 에서 300 걸음 걸은 그 일이 이 한 줄

print(f"    w = {model.coef_[0]:.4f}  b = {model.intercept_:.4f}")

# 규칙 3: 학습해서 얻은 값은 끝에 밑줄(_). coef_ = w 들, intercept_ = b
print("    01 에서 걸어서 찾은 값: w = 0.9840, b = 14.7799  ← 같죠?")

# ③ 예측. 새 데이터도 '표 모양'이라 대괄호 두 겹 [[ ]]
print("    공기온도 300 → 예측:", round(model.predict([[300.0]])[0], 2))

# .score = 회귀면 R², 분류면 정확도
# 사이킷런 LinearRegression은 걷지 않고 '공식'으로 단 번에 문제를 해결 -> lr, epoch 없음
# 선형회귀(와 Ridge)는 공식이 있고, 로지스틱 회귀부터는 사이킷런도 속으로 걸음
# max_iter(알고리즘이 최적의 해를 찾기 위해 수행하는 최대 반복 횟수)가 그 흔적
print("    R²:", round(model.score(X1, y), 4), "(01 의 0.7989)")

# 가장 흔한 에러 확인 - X를 한 줄(1차원)로 넣은 경우
try:
    LinearRegression().fit(x, y)
except ValueError as e:
    print("    [에러] ", str(e).splitlines()[0], "→ reshape(-1, 1) 하라는 뜻")

# [1] x.shape (200,) → X1.shape (200, 1)
#     w = 0.9840  b = 14.7799
#     01 에서 걸어서 찾은 값: w = 0.9840, b = 14.7799  ← 같죠?
#     공기온도 300 → 예측: 309.97
#     R²: 0.7989 (01 의 0.7989)
#     [에러]  Expected 2D array, got 1D array instead: → reshape(-1, 1) 하라는 뜻


# =====================================================================
# 2. 나누기 + 표준화 + 다변수 회귀 — 02 를 도구로
# =====================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[특징이름].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# 02의 '섞고 70:30으로 자르기'가 이 한 줄
# 돌려주는 순서(X학습, X시험, y학습, y시험) -> 순서가 틀리면 전체 결과가 틀어짐!
# random_state=42는 02의 RandomState(42)와 동일한 역할(재현성)
# 단 섞는 방식이 달라 02와 '같은 140대'는 아님에 주의

print("\n[2] train", X_train.shape, "/ test", X_test.shape)

scaler = StandardScaler()  # 표준화 도구

scaler.fit(X_train)  # 학습용의 평균·표준편차를 '외운다'  (02 의 mu, sd 계산)
# scaler.fit(X_test)를 하게 되면 누수!

Z_train = scaler.transform(X_train)  # 외운 값으로 변환  (02 의 (X − mu) / sd)

Z_test = scaler.transform(X_test)  # 시험용도 학습용 눈금으로 변환만

print(
    "    scaler 가 외운 평균:",
    scaler.mean_.round(1),
    "← 02 의 mu 역할 (분할이 달라 값은 조금 다름)",
)

reg = LinearRegression().fit(Z_train, y_train)
# 만들기와 fit 을 한 줄에 붙여 쓰기도 합니다


# 표준화 가중치를 보면 각 변수가 결과에 얼마나, 어느 방향으로 영향을 주는 지 서로 비교 가능
# dict(zip(이름, 값)) = {이름:값} 짝 딕셔너리
print("    표준화 가중치:", dict(zip(특징이름, reg.coef_.round(3).tolist())))
# 02와 분할이 달라 숫자는 조금 다르지만 그림은 동일함
# 공기온도가 압도적(+1.99), 나머지 거의 0

print(
    f"    train R² {reg.score(Z_train, y_train):.4f} / test R² {reg.score(Z_test, y_test):.4f}"
)
# train 0.816 / test 0.143 -> 차이 0.073
# 02의 경보선(0.005 정상 / 0.10 의심) 사이
# 시험용이 60대뿐(적음)이라 흔들리는 것으로 해석하면 문제 없음
# 진짜 과적합에 해당하는지 확인하려면 교차 검증

# [2] train (140, 4) / test (60, 4)
#     scaler 가 외운 평균: [ 300.1 1494.9   41.2  117.2] ← 02 의 mu 역할 (분할이 달라 값은 조금 다름)
#     표준화 가중치: {'공기온도': 1.986, '회전수': 0.004, '토크': -0.078, '공구마모': -0.008}
#     train R² 0.8155 / test R² 0.7425


# =====================================================================
# 3. Pipeline — "표준화는 학습용으로만" 을 도구가 대신 지키게
# =====================================================================
from sklearn.pipeline import make_pipeline

# 파이프라인 : 전처리(표준화 등)와 모델을 순서대로 이어 붙여 '하나의 모델'처럼 다루는 것
# fit을 할 때엔 학습용으로 스케일러를 맞추고
# predict를 할 때에는 변환만 하는 것을 도구가 알아서 해줌
# 사람이 실수로 scaler.fit(X_test)를 하지 않도록 구조로 방지하는 것
# 새 데이터도 '원래 눈금' 그대로 넣으면 됨

# 왼쪽부터 순서대로 : 표준화 -> 선형회귀
pipe = make_pipeline(StandardScaler(), LinearRegression())

# 원래 눈금 X 를 그냥 넣는다. 안에서 표준화까지 함
pipe.fit(X_train, y_train)
print("\n[3] Pipeline test R²:", round(pipe.score(X_test, y_test), 4), "(위와 같음)")
print(
    "    새 설비 [공기 300, 회전 1500, 토크 40, 마모 100] → 공정온도",
    round(pipe.predict([[300, 1500, 40, 100]])[0], 2),
)

# [3] Pipeline test R²: 0.7425 (위와 같음)
#     새 설비 [공기 300, 회전 1500, 토크 40, 마모 100] → 공정온도 310.03

# 함정 누수(leakage)가 얼마나 속이는 지 눈으로 확인
# 누수 : 시험용 데이터의 정보가 학습 쪽으로 새어 들어가는 것
#        점수가 부풀려지고 현장에서 무너지므로 주의
# 제일 흔한 사고 : 00에서 배운 '중복 행 제거'를 잊은 경우
# 같은 측정이 두 번 들어 있으면 무작위로 나눌 때 쌍둥이가 학습용과 시험용에 갈라져 들어감
# 모델은 시험 문제를 이미 학습용에서 본 셈이 되는 것


# KNeighborsRegressor(1) : 제일 비슷한 설비 한 대를 찾아 그 설비의 정답을 그대로 말하는 단순
# 외우기에 특화되어 있어 누수를 드러내기 좋음
from sklearn.neighbors import KNeighborsRegressor

이웃 = lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=1))

# 일부러 모든 행을 두 번씩 (중복 제거를 깜빡한 상황)
X중복 = np.vstack([X, X])
y중복 = np.concatenate([y, y])
d_tr, d_te, dy_tr, dy_te = train_test_split(
    X중복, y중복, test_size=0.3, random_state=42
)
print("\n[3-1] 중복 행을 안 지우고 나누면 (누수)")
print(
    f"    중복 있음 test R² {이웃().fit(d_tr, dy_tr).score(d_te, dy_te):.4f}   <- 훌륭해 보인다"
)
print(
    f"    중복 없음 test R² {이웃().fit(X_train, y_train).score(X_test, y_test):.4f}   <- 이게 이 모델의 진짜 실력"
)
print(
    "    같은 모델, 같은 데이터입니다. 중복을 안 지운 것 하나로 점수가 네 배가 됐어요."
)

# [3-1] 중복 행을 안 지우고 나누면 (누수)
#     중복 있음 test R² 0.8177   <- 훌륭해 보인다
#     중복 없음 test R² 0.1864   <- 이게 이 모델의 진짜 실력
#     같은 모델, 같은 데이터입니다. 중복을 안 지운 것 하나로 점수가 네 배가 됐어요.

# 현장에서 중복 제거를 하지 않고 학습 시험을 본 경우
# '개발할 때에는 0.9가 나왔는데 현장에 적용하니 0.4'라는 대형 사고가 발생할 가능성 존재
# 대부분의 원인
# ex) 데이터를 늘리려고 중복 제거 하지 않음, 같은 설비를 여러번 잰 것을 다른 데이터로 취급

# 현장에 넣으면 나오는 점수 -> 0.19
# 0.82는 존재하지 않는 실력이었던 것

# 개발할 때는 잘 나왔는데 현장에서는 안 된다는 판단을 내리는 것 <- 절대 금지

# 이러한 상황의 발생을 방지하기 위한 방법
# 1) 나누기 전에 중복을 지움
#    같은 설비, 같은 시각이 두 번 있으면 의심할 것
# 2) 표준화 같은 전처리는 반드시 학습용으로만 fit 할 것
#    이것을 자동으로 지켜 주는 것이 위의 pipeline
# 현장 배포 형태가 바로 이것
# 센서 원값 4개를 넣으면 답이 나오는 '한 덩어리', 실무 코드는 거의 이러한 형태


# =====================================================================
# 4. 분류 — 03 의 로지스틱 회귀를 도구로
# =====================================================================
from sklearn.linear_model import LogisticRegression

yc = df["고장여부"].values  # 분류 정답: 0/1

# stratify=yc = 고장 비율을 양쪽에 똑같이. 03 2번에서 손으로 한 일이 단어 하나
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, yc, test_size=0.3, random_state=3, stratify=yc
)

print(
    "\n[4] 분류 — 학습용 고장", yc_train.sum(), "대 / 시험용 고장", yc_test.sum(), "대"
)

# max_iter = 최대 걸음 수. 기본 100 이 모자라면 경고가 떠서 1000 으로
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

# 03 의 sigmoid + 로그손실 + 경사하강 2000 바퀴가 이 한 줄
clf.fit(Xc_train, yc_train)

# predict_proba() : (60, 2)표
#                   열 0은 정상일 확률, 열 1은 고장일 확률, [:,1]은 모든 행의 1번 열
p = clf.predict_proba(Xc_test)[:, 1]

# 0.5 기준 판정, 임계값을 바꾸려면 p를 직접 자름(p >= 0.2)
판정 = clf.predict(Xc_test)

# 정렬 → [::-1] 뒤집어 큰 순 (00 미리보기 ⑥) → 앞 5개
print("    시험용 고장 확률 상위 5:", np.sort(p)[::-1][:5].round(3))

print(
    "    정확도:",
    round(clf.score(Xc_test, yc_test), 3),
    "← 03 에서 배웠듯 이것만 보면 속는다",
)

# [4] 분류 — 학습용 고장 6 대 / 시험용 고장 2 대
#     시험용 고장 확률 상위 5: [0.366 0.348 0.243 0.24  0.234]
#     정확도: 0.967 ← 03 에서 배웠듯 이것만 보면 속는다

# 03 손코드의 1등 확률은 0.528이었는데 여기서는 0.366으로 낮게 나옴
# 사이킷런 LogisticRegression은 기본적으로 '규제'가 켜져 있음
# 가중치를 살짝 누르기 때문에 확률이 덜 극단적으로 나오는 것
# 분할도 마찬가지로 다름
# 값은 달라도 '고장 확률이 전반적으로 낮게 나온다'는 그림은 동일함


# =====================================================================
# 5. 채점 도구 — 03 에서 손으로 센 네 칸·재현율을 함수로
# =====================================================================
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score,
)

print(
    "\n[5] 혼동행렬 (사이킷런은 [[정상→정상, 정상→고장], [고장→정상, 고장→고장]] 순서)"
)

# 인자 순서: (실제, 판정). 바꾸면 표가 뒤집힙니다
print(confusion_matrix(yc_test, 판정))

# ravel() : 2*2 표를 한 줄 4개로 폄(왼쪽 위 부터 TN, FP, FN, TP)
tn, fp, fn, tp = confusion_matrix(yc_test, 판정).ravel()

print(f"    잡음(TP) {tp}  놓침(FN) {fn}  헛경보(FP) {fp}  통과(TN) {tn}")

print("\n    classification_report — 정밀도·재현율을 한 표에 (고장 행을 보세요)")
print(
    classification_report(yc_test, 판정, target_names=["정상", "고장"], zero_division=0)
)
# 읽는 법
# 행 = 편(정상/고장)
# precision(정밀도), recall(재현율), f1-score(둘의 조화평균), support(그 편의 실제 대수)
# '고장' 행 : recall 0.00 -> 고장 2대 중 0대 잡음 -> 이 한 칸이 제조에서 제일 중요한 숫자
# '정상' 행이 아무리 좋아도 소용 없음 -> 아래 accuracy 0.97은 03에서 학습한 거짓말
# zero_division=0 : 0으로 나눌 때 경고 대신 0을 쓰라는 뜻(고장을 하나도 안 잡으면 정미로 분모 0)

print("    임계값을 내리면 (03 8번을 도구로)")
for th in [0.5, 0.3, 0.2, 0.1]:
    판정_th = (p >= th).astype(int)
    print(
        f"      임계값 {th}: 재현율 {recall_score(yc_test, 판정_th, zero_division=0):.2f}"
        f"  정밀도 {precision_score(yc_test, 판정_th, zero_division=0):.2f}"
    )
# 0.5, 0.3에서는 전멸
# 0.1까지는 내려야 둘 다 잡아내고 있음
# 재현율(Recall, 실제로 고장난 것 중에서 모델이 고장이라고 찾아낸 비율)
# 정밀도(Precision, 모델이 고장이라고 말한 것 중 진짜 고장인 비율)
# 재현율 1.0, 정밀도 0.17 -> 고장이라고 한 12대 중 진짜 고장은 2대
# 처방 1) 임계값 내리기 -> 학습은 그대로 두고 자르는 선만 옮김
# 처방 2) 학습할 때부터 '고장 1대를 정상 약 22대만큼 무겁게 세라'고 시키기
# 학습용에 정상134:고장6 -> 134/6 ~= 22배 -> 사이킷런이 이 비율을 자동으로 계산해 벌점에 곱함
clf_b = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
)
clf_b.fit(Xc_train, yc_train)
판정_b = clf_b.predict(Xc_test)
print(
    f"    class_weight='balanced' 로 다시 학습 → 임계값 0.5 에서 재현율 {recall_score(yc_test, 판정_b):.2f}"
    f"  정밀도 {precision_score(yc_test, 판정_b, zero_division=0):.2f}  (놓침 대신 헛경보를 택한 것)"
)

# [5] 혼동행렬 (사이킷런은 [[정상→정상, 정상→고장], [고장→정상, 고장→고장]] 순서)
# [[58  0]
#  [ 2  0]]
#     잡음(TP) 0  놓침(FN) 2  헛경보(FP) 0  통과(TN) 58

#     classification_report — 정밀도·재현율을 한 표에 (고장 행을 보세요)
#               precision    recall  f1-score   support

#           정상       0.97      1.00      0.98        58
#           고장       0.00      0.00      0.00         2

#     accuracy                           0.97        60
#    macro avg       0.48      0.50      0.49        60
# weighted avg       0.93      0.97      0.95        60

#     임계값을 내리면 (03 8번을 도구로)
#       임계값 0.5: 재현율 0.00  정밀도 0.00
#       임계값 0.3: 재현율 0.00  정밀도 0.00
#       임계값 0.2: 재현율 0.50  정밀도 0.20
#       임계값 0.1: 재현율 1.00  정밀도 0.17
#     class_weight='balanced' 로 다시 학습 → 임계값 0.5 에서 재현율 1.00  정밀도 0.15  (놓침 대신 헛경보를 택한 것)

# 어느 처방을 사용해야 하는가
# 임계값 조정 :
# 이미 학습된 모델을 두고 현장에서 '선'만 옮길 때 -> 비용이 바뀌면 선만 다시 긋기
# class_ weight : 학습 자체가 소수편(고장)을 무시할 때 -> 모델이 고장 패턴을 '배우도록' 강제 -> 재학습 필요
# 실무에서는 보통 두 가지 방법을 다 사용
# balanced로 학습하고 임계값은 비용을 확인한 후 최종적으로 조정


# =====================================================================
# 6. 규제 — 과적합 처방을 옵션 하나로 (Ridge)
# =====================================================================
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# 규제 : 가중치가 너무 커지지 않게 손실에 '벌점'을 더하는 것
#       과적합(외우기)을 누름
# 02의 5-1에서 과적합 모델은 가중치가 커졌던 것을 확인 -> 그것을 억누르면 외우기가 어려워짐

# Ridge : 벌점을 붙인 선형회귀
#         alpha가 별점 세기(클수록 셈), 0이면 그냥 LinearRegression
# 센서는 많고 데이터는 적을 때 모델이 학습 데이터를 통째로 외우는 것을 막기 위해 사용

# Lasso : 벌점 방식이 달라 쓸모없는 가중치를 아예 0으로 만들기(센서 고르기 효과)

print("\n[6] Ridge alpha 별 점수 — 이 데이터(센서 4개)에선")
for a in [0.01, 1, 10, 100]:
    r = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X_train, y_train)
    print(
        f"    alpha={a:<5} → train {r.score(X_train, y_train):.4f}  test {r.score(X_test, y_test):.4f}"
    )
# alpha=10 까지는 거의 그대로
# alpha=100이면 너무 눌러 결과가 너무 나빠짐 -> 억지로 과소 적합을 만들어 버린 것
# 이 데이터는 손잡이가 5개뿐이라 외울 힘이 없어 규제가 할 일이 없음

# 규제가 진짜 필요한 상황을 만들어 볼 것
# 센서 4개를 서로 곱하고 제곱하여 특징 70개로 부풀리면 외울 힘이 생김
print(
    "\n[6-1] 특징을 70개로 부풀리면 (PolynomialFeatures) — 과적합이 생기고, Ridge 가 고친다"
)
for 이름, 모델 in [
    ("규제 없음(LinearRegression)", LinearRegression()),
    ("Ridge(alpha=10)", Ridge(alpha=10)),
]:
    poly = make_pipeline(
        StandardScaler(), PolynomialFeatures(4), StandardScaler(), 모델
    ).fit(X_train, y_train)
    print(
        f"    {이름:26s} → train {poly.score(X_train, y_train):.3f}  test {poly.score(X_test, y_test):.3f}"
    )

# [6] Ridge alpha 별 점수 — 이 데이터(센서 4개)에선
#     alpha=0.01  → train 0.8155  test 0.7425
#     alpha=1     → train 0.8155  test 0.7438
#     alpha=10    → train 0.8116  test 0.7500
#     alpha=100   → train 0.6763  test 0.6357

# [6-1] 특징을 70개로 부풀리면 (PolynomialFeatures) — 과적합이 생기고, Ridge 가 고친다
#     규제 없음(LinearRegression)    → train 0.892  test 0.430
#     Ridge(alpha=10)            → train 0.869  test 0.649

# 규제 없음
# train 0.89 / test 0.43 -> 차이 0.46 -> 전형적인 과적합
# 학습 점수는 올랐는데 시험 점수가 무너짐

# 규제 있음(Ridge 추가)
# Ridge 10 : train 0.87 / test 0.65
# 학습 점수를 조금 양보하고 시험 점수를 되찾음

# 원래 4개 특징(0.74)보다는 못함
# 애초에 이 데이터는 직선이 정답이기 때문에 부풀릴 이유가 없었던 것
# 따라서 train >> test인 경우에는 규제(alpha 올리기), 데이터 더 모으기, 특징 줄이기 필요
# train과 test 둘 다 낮은 경우에는 alpha 줄이기, 특징을 더 모으기, 더 유연한 모델 사용


# =====================================================================
# 7. 손잡이(alpha 등) 고르기 — test 를 훔쳐보지 않고: 교차검증
# =====================================================================
# 하이퍼 파라미터 : 학습으로 정해진 것이 아니라 '사람이 정하는' 손잡이
#                 alpha, 학습률, 에폭, 임계값 등
# test 점수를 보며 alpha를 고르면 그 test 점수는 이미 '고르는 데 쓴 점수'라 부풀려짐

# 교차검증(CV) : 학습용을 5조각으로 나눔 -> 4조각으로 배우고 1조각으로 채점 -> 5번 반복 -> 평균
# test는 건드리지 않음

from sklearn.model_selection import cross_val_score, GridSearchCV

cv = cross_val_score(
    make_pipeline(StandardScaler(), LinearRegression()), X_train, y_train, cv=5
)

# 조각마다 0.71~0.85로 흔들림
# train / test 차이인 0.073이 이 흔들림 안에 있는 것
# 과적합이 아님에 주의! -> 데이터가 적어서 튀는 것
# 이렇게 교차 검증은 '그 차이가 진짜 과적합인지, 우연인지'를 가려줌
print("\n[7] 교차검증 5조각 R²:", cv.round(3), "→ 평균", round(cv.mean(), 4))

# GridSearchCV : 후보값들을 전부 교차검증으로 돌려 최고를 고르고 그 값으로 다시 학습해주는 도구
grid = GridSearchCV(
    make_pipeline(StandardScaler(), Ridge()),
    # 단계이름__옵션 : 파이프라인 안에 있는 ridge의 alpha
    {"ridge__alpha": [0.01, 0.1, 1, 10, 100]},
    cv=5,
)

# 후보 5개 × 5조각 = 25번 학습
grid.fit(X_train, y_train)

print(
    "    교차검증으로 고른 alpha:",
    grid.best_params_,
    "/ CV 평균 R²:",
    round(grid.best_score_, 4),
)
print("    그 모델의 test R² (이제 딱 한 번):", round(grid.score(X_test, y_test), 4))

# [7] 교차검증 5조각 R²: [0.723 0.848 0.846 0.71  0.824] → 평균 0.7901
#     교차검증으로 고른 alpha: {'ridge__alpha': 1} / CV 평균 R²: 0.7904
#     그 모델의 test R² (이제 딱 한 번): 0.7438

# 지금까지 계속 손으로 직접 작성한 이유
# 에러, 점수 이상한 경우 모델들 안이 어떻게 생겼는지까지는 알 수 있음
# 순서가 매우 중요!
# 학습용 안에서 교차검증으로 손잡이를 고르고 마지막에 test를 딱 한 번 보는 것


# =====================================================================
# 8. 저장 — 내일 다시 쓰려면
# =====================================================================
# joblib : 대용량 데이터 처리, 머신러닝 모델 저장(직렬화), 병렬 컴퓨팅등을 구현하는 라이브러리
import joblib

# 파이프라인(스케일러+모델) 통째로 파일로. 학습 결과가 다 들어감
joblib.dump(clf, "고장분류기.joblib")

# 내일 이 한 줄로 불러오면 재학습 없이 바로 예측
다시 = joblib.load("고장분류기.joblib")

print(
    "\n[8] 저장 후 불러와 예측 — 새 설비 [공기 299, 회전 1400, 토크 55, 마모 240] 고장 확률:",
    round(다시.predict_proba([[299, 1400, 55, 240]])[0, 1], 3),
)

# 수업 폴더를 깨끗하게 (실제론 남겨 둡니다)
# os.remove("고장분류기.joblib")

# 파이프라인 전체를 저장하는 이유
# 스케일러가 외운 평균, 표준편차까지 함께 저장되어야 새 데이터를 같은 눈금으로 변환
# 모델만 저장하면 표준화를 다르게 해 엉뚱한 답이 나옴


# =====================================================================
# 9. 정리 — 손코드 ↔ 사이킷런 대응표, 그리고 언제 쓰나
# =====================================================================
print("""
[9] 손으로 한 것 ↔ 사이킷런
    01 표준화 (x−평균)/표준편차     ↔  StandardScaler().fit(train) / .transform()
    02 섞어서 70:30 자르기          ↔  train_test_split(X, y, test_size=0.3, random_state=…, stratify=y)
    01·02 경사하강 300·500걸음      ↔  LinearRegression().fit(X, y)   (공식으로 단번에)
    03 sigmoid + 로그손실 + 경사하강 ↔  LogisticRegression().fit(X, y)  (속으로 걸음, max_iter)
    03 확률 / 0.5 판정              ↔  .predict_proba(X)[:, 1] / .predict(X)
    03 네 칸·재현율 손으로 세기      ↔  confusion_matrix / classification_report / recall_score
    w, b                           ↔  .coef_, .intercept_
    (없음) 과적합 처방               ↔  Ridge(alpha) / Lasso(alpha)
    (없음) 불균형 처방               ↔  class_weight="balanced" (+ 임계값 조정)
    (없음) 손잡이 고르기             ↔  cross_val_score / GridSearchCV

    사이킷런은 언제?  표 데이터, 몇백~몇십만 행, 선형모델·트리·부스팅 → 실무 기본. fit 한 줄. 이 과정 데이터는 전부 여기.
    파이토치는 언제?  학습 루프를 내 손으로 쥐어야 할 때 — 아주 큰 데이터, 손실을 내 맘대로 바꿀 때,
                     그리고 다음 과정에서 배울 '신경망'(층 쌓기) → 05 파일에서 같은 문제를 파이토치로 풀어 봅니다.
""")

# =====================================================================
# 실습 — Ridge 말고 Lasso 는?
# =====================================================================
# [문제] 9번 대응표에 Ridge 와 함께 Lasso 가 적혀 있었습니다.
# Lasso(alpha=...) 로 바꿔서 alpha 0.01 / 0.1 / 1.0 의 train, test 점수를 보세요.
#
#   힌트: from sklearn.linear_model import Lasso 부터. 나머지는 6번의 Ridge 코드와 같습니다.
from sklearn.linear_model import Lasso

print("\n[실습] Lasso alpha 별 점수 — 이 데이터(센서 4개)에선")
for a in [0.01, 0.1, 1.0]:
    la = make_pipeline(StandardScaler(), Lasso(alpha=a)).fit(X_train, y_train)
    print(
        f"alpha={a:<5} → train {la.score(X_train, y_train):.4f}  test {la.score(X_test, y_test):.4f}"
    )

# [실습] Lasso alpha 별 점수 — 이 데이터(센서 4개)에선
# alpha=0.01  → train 0.8155  test 0.7437
# alpha=0.1   → train 0.8122  test 0.7486
# alpha=1.0   → train 0.6083  test 0.5910 -> 점수 차이보다는 성능이 크게 하락

# Ridge에서는 alpha=100인 경우 과소 적합
# Lasso에서는 alpha=1.0인 경우 과소 적합

# Ridge는 가중치를 조금씩 부드럽게 줄이는 방식 -> 필요한 변수의 영향력이 어느 정도 남아있음
# Lasso는 규제가 강해지면 가중치를 줄이다가 아예 0으로 만들어버리기도 함
# -> Ridge의 alpha=1과 Lasso의 alpha=1은 같은 강도의 규제가 아님에 주의!
# -> Ridge와 Lasso는 규제 방식이 다르기 때문에 같은 alpha값이라도 규제 효과가 같지 않음
