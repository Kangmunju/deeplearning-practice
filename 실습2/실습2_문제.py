# =====================================================================
#  실습 2 — 진동으로 압연 모터 전류를 맞혀 보기 (02 다변수 선형회귀)
# =====================================================================
#  실행: python 실습2_문제.py      (이 파일이 있는 폴더에서)
#  필요한 것: numpy, pandas  /  데이터는 옆의 데이터/ 폴더에 들어 있습니다.
#
#  [상황]
#    P제철 열간압연기(SPM01)에 진동센서 두 개(TOP·BOT)와 모터 전류계가 붙어 있습니다.
#    그런데 전류계가 자주 고장 납니다. 진동만 있을 때 전류를 추정할 수 있을까요?
#    맞힐 대상(정답) = CUR-MTR_RMS  (모터 전류의 실효값)
#
#  [쓰는 도구]  02 에서 배운 것 전부. 새 라이브러리 없습니다.
#    다변수 X / train·test 분할 / 열별 표준화(학습용 통계로만) / 경사하강 / R2
#    ※ 02_선형회귀_다변수_train_test.py 를 옆에 띄워 놓고 베껴 쓰세요. 그게 정상입니다.
#
#  [푸는 법]  TODO 를 위에서부터 하나씩 채우고, 그때그때 실행해서 숫자를 확인하세요.
#             한 번에 다 짜고 실행하면 어디서 틀렸는지 못 찾습니다.
# =====================================================================

import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "데이터")
d = pd.read_csv(os.path.join(DATA, "T-CR1-SPM01_압연특징.csv"), encoding="utf-8-sig")
# ↑ encoding="utf-8-sig" 빠뜨리면 첫 열 이름이 깨져서 KeyError 납니다.


# =====================================================================
# A. 데이터부터 본다  (모델 얘기는 아직 이르다)
# =====================================================================
# [A1] 표의 모양과 열 이름, 결측 개수를 찍으세요.
#      힌트: d.shape / list(d.columns) / d.isna().sum().sum()
# TODO
print("A. 데이터부터 본다")
print("d.shape :", d.shape)  # d.shape : (570, 19)
print("list(d.columns) :", list(d.columns))
# list(d.columns) : ['MEAS_DT', 'BURST_ID', 'ROWS', 'DUR_S', 'VIB-TOP_RMS', 'VIB-TOP_KUR', 'VIB-TOP_CRF', 'VIB-TOP_PTP', 'VIB-TOP_STD', 'VIB-BOT_RMS', 'VIB-BOT_KUR', 'VIB-BOT_CRF', 'VIB-BOT_PTP', 'VIB-BOT_STD', 'CUR-MTR_RMS', 'CUR-MTR_KUR', 'CUR-MTR_CRF', 'CUR-MTR_PTP', 'CUR-MTR_STD']
print("d.isna().sum().sum() :", d.isna().sum().sum())  # d.isna().sum().sum() : 0


# [A2] 정답으로 쓸 CUR-MTR_RMS 의 요약통계를 보세요. (describe)
#      → 이 값이 대략 몇에서 몇 사이인지 말할 수 있어야 합니다.
#        나중에 "MSE 500" 이 큰 건지 작은 건지 판단하는 기준이 됩니다.
# TODO
print("CUR-MTR_RMS 의 요약통계 : ", d["CUR-MTR_RMS"].describe())
# std       38.382288
# min        6.325006
# 25%       82.244598
# 50%       93.381820
# 75%      145.054399
# max      193.420110
# Name: CUR-MTR_RMS, dtype: float64


# [A3] 숫자 열들의 상관계수 중, CUR-MTR_RMS 와의 상관만 크기순으로 보세요.
#      힌트: d.corr(numeric_only=True)["CUR-MTR_RMS"].sort_values()
#
#      ★ 보고 나서 답하세요 (주석으로 적어 두기) ★
#        (1) 상관이 0.98, 0.99 로 말도 안 되게 높은 열이 몇 개 보입니다. 이름이 뭔가요?
#        (2) 그 열들을 입력으로 쓰면 안 되는 이유가 있습니다. 뭘까요?
#            힌트: 열 이름의 앞부분을 보세요. CUR-MTR-... 로 시작하죠.
#                 전류계가 고장 나서 전류를 추정하려는 건데, 그 입력은 어디서 옵니까?
#      내 답: (1) CUR-MTR_PTP    0.982105 / CUR-MTR_STD    0.998795
#             (2) 전류계가 고장나면 위 두 개의 값도 못 얻음, 실제 예측 모델의 입력으로 사용을 못 한다.
# TODO
print("CUR-MTR_RMS 와의 상관만 크기순")
print(d.corr(numeric_only=True)["CUR-MTR_RMS"].sort_values())
# CUR-MTR_RMS 와의 상관만 크기순
# VIB-BOT_KUR   -0.469390
# VIB-BOT_CRF   -0.448028
# CUR-MTR_KUR   -0.249195
# CUR-MTR_CRF   -0.089699
# VIB-TOP_STD    0.082713
# VIB-TOP_RMS    0.093488
# VIB-TOP_PTP    0.142816
# ROWS           0.229055
# DUR_S          0.229055
# VIB-TOP_KUR    0.230738
# VIB-TOP_CRF    0.327160
# VIB-BOT_PTP    0.684858
# VIB-BOT_STD    0.712887
# VIB-BOT_RMS    0.738609
# CUR-MTR_PTP    0.982105
# CUR-MTR_STD    0.998795
# CUR-MTR_RMS    1.000000
# Name: CUR-MTR_RMS, dtype: float64

# =====================================================================
# B. 입력 고르고 train / test 나누기
# =====================================================================
# [B1] 입력(특징) 4개를 아래 이름 그대로 쓰세요. 정답은 CUR-MTR_RMS.
feature_name = ["VIB-BOT_RMS", "VIB-BOT_PTP", "VIB-BOT_KUR", "VIB-TOP_RMS"]
X = d[feature_name].values.astype(float)
y = d["CUR-MTR_RMS"].values.astype(float)
# X.shape, y.shape 를 찍어서 (570, 4) 와 (570,) 인지 확인하세요.
# TODO
print("B. 입력 고르고 train / test 나누기")
print("X.shape :", X.shape)  # X.shape : (570, 4)
print("y.shape :", y.shape)  # y.shape : (570,)


# [B2] 7:3 으로 나누세요. 반드시 '섞은 다음에' 자릅니다.
#      RandomState(42) 를 쓰면 정답지와 숫자가 똑같이 나옵니다.
#      힌트: 순서 = np.random.RandomState(42).permutation(len(X))
#            n_train = int(len(X) * 0.7)
#      학습용 몇 대 / 시험용 몇 대인지 찍으세요.
# TODO
sequence = np.random.RandomState(42).permutation(len(X))
n_train = int(len(X) * 0.7)

train_idx = sequence[:n_train]
test_idx = sequence[n_train:]

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

print(f"학습용 {len(train_idx)} / 시험용 {len(test_idx)}")
# 학습용 399 / 시험용 171


# [B3] 열별 표준화. ★ mu 와 sd 는 학습용에서만 구합니다 ★
#      시험용도 학습용의 mu, sd 로 변환하세요.
#      확인: 표준화 후 학습용 열별 평균은 0, 퍼짐은 1.
#            시험용 평균은 0 이 아닙니다. 그게 맞습니다 (이유를 말할 수 있어야 합니다).
# TODO
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)

Z_train = (X_train - mu) / sd
Z_test = (X_test - mu) / sd

print("표준화 후 학습용 열별 평균:", Z_train.mean(axis=0).round(3))  # [0. 0. 0. 0.]
print("표준화 후 학습용 열별 퍼짐:", Z_train.std(axis=0).round(3))  # [1. 1. 1. 1.]
print("시험용 열별 평균:", Z_test.mean(axis=0).round(3))
# [-0.09  -0.089  0.128 -0.093]
# 시험용 데이터는 자신의 평균과 표준편차로 표준화한 게 아니기 때문에 0이 안 나옴.


# =====================================================================
# C. 학습 — 02 의 함수를 그대로 가져다 쓰세요
# =====================================================================
# [C1] 예측 / 손실 / 기울기_밟아보기 / 학습 / R2 / MSE 를 02 에서 복사해 오세요.
#      한 글자도 안 바꿔도 됩니다. 그게 이 실습의 포인트입니다.
#      (데이터가 바뀌어도 걷는 방법은 안 바뀝니다)
# TODO
print("C. 학습")
h = 0.0001


def 예측(Z, w, b):
    return Z @ w + b


def 손실(Z, y, w, b):  # 01 과 완전히 같음: (실제 − 예측)² 의 평균 = MSE
    return np.mean((y - 예측(Z, w, b)) ** 2)


def 기울기_밟아보기(Z, y, w, b):
    gw = np.zeros(len(w))
    for j in range(len(w)):  # j = 0,1,2,3 : j 번째 손잡이만 움직여 본다
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[j] += h  # j 번째만 살짝 키우고
        w_minus[j] -= h  # j 번째만 살짝 줄여서
        gw[j] = (손실(Z, y, w_plus, b) - 손실(Z, y, w_minus, b)) / (
            2 * h
        )  # (오른쪽 손실 − 왼쪽 손실) ÷ 거리
    gb = (손실(Z, y, w, b + h) - 손실(Z, y, w, b - h)) / (2 * h)  # b 도 한 번
    return gw, gb


def 학습(Z, y, lr=0.1, epochs=500):
    w = np.zeros(Z.shape[1])  # Z.shape[1] = 열 수 = 4. 가중치 4개를 0 에서 출발
    b = 0.0
    for _ in range(epochs):  # 500 바퀴 (학습 횟수)
        gw, gb = 기울기_밟아보기(Z, y, w, b)
        w = w - lr * gw  # 경사 하강 4개가 한꺼번에 내리막 쪽으로 보폭 lr만큼
        b = b - lr * gb
    return w, b


def R2(y, yhat):  # 01 9번의 R². 1 에 가까울수록 좋음, 0 = 평균만 말하는 수준
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def MSE(y, yhat):  # 손실과 같은 식. 채점용으로 이름만 따로
    return np.mean((y - yhat) ** 2)


# [C2] 학습용으로 학습(lr=0.1, epochs=500)하고, 특징별 가중치와 절편 b 를 찍으세요.
# TODO
w, b = 학습(Z_train, y_train, lr=0.1, epochs=500)
for name, wi in zip(feature_name, w):
    print(f"{name:6s} w = {wi:+.3f}")
print(f"절편 b = {b:.3f}")
# VIB-BOT_RMS w = +24.692
# VIB-BOT_PTP w = +14.966
# VIB-BOT_KUR w = -5.908
# VIB-TOP_RMS w = -18.962
# 절편 b = 112.640

# [C3] 학습용 R2 / 시험용 R2 를 나란히 찍으세요. MSE 도 같이.
#
#      ★ 답하세요 ★
#        차이가 얼마입니까? 02 의 경보선(0.05 정상 / 0.10 넘으면 의심)에 비춰 보면
#        이 모델은 건강한가요, 과적합인가요?
#        ai4i 데이터(02)에서는 차이가 0.03 이었습니다. 왜 여기선 다를까요?
#      내 답: R2 차이 0.1903 -> 0.1보다 큰 차이 -> 과적합 -> 데이터가 달라서?
# TODO
tr_pred = 예측(Z_train, w, b)  # 학습용
te_pred = 예측(Z_test, w, b)  # 시험용
tr_r2 = R2(y_train, tr_pred)
te_r2 = R2(y_test, te_pred)
tr_mse = MSE(y_train, tr_pred)
te_mse = MSE(y_test, te_pred)
r2_gap = tr_r2 - te_r2

print(f"학습용 R2 {round(tr_r2, 4)} / 시험용 R2 {round(te_r2, 4)}")
# 학습용 R2 0.7982 / 시험용 R2 0.608
print(f"학습용 MSE {round(tr_mse, 4)} / 시험용 MSE {round(te_mse, 4)}")
# 학습용 MSE 307.9932 / 시험용 MSE 519.8969
print(f"R2 차이 {round(r2_gap, 4)}")
# R2 차이 0.1903


# =====================================================================
# D. 함정 1 — "점수가 너무 좋으면 의심하라"
# =====================================================================
# [D1] 특징에 "CUR-MTR_STD" 를 하나 추가해서(총 5개) 다시 학습하고 채점하세요.
#      B2 의 분할(순서)은 그대로 재사용합니다. 표준화는 다시 해야 합니다(열이 5개니까).
#
#      ★ 답하세요 ★
#        (1) R2 가 몇으로 나왔나요? 학습용·시험용 둘 다 적으세요.
#            학습용 R2 0.9984 / 시험용 R2 0.9982
#        (2) 이 모델을 현장에 넣으면 잘 될까요? 이유는?
#            현장에서 쓰기 어려울 것 같음.
#            추가한 특징도 전류계에서 나온 값이라 고장나면 이 입력값 자체를 얻지 못함.
#        (3) 이걸 부르는 이름이 있습니다 — '누수(leakage)'.
#            이 경우 정확히 무엇이 새어 들어온 겁니까?
#            RMS와 매우 가까운 전류계 정보인 STD가 입력 특징으로 들어간 것.
#
# TODO
print("D. 함정")
new_feature_name = [
    "VIB-BOT_RMS",
    "VIB-BOT_PTP",
    "VIB-BOT_KUR",
    "VIB-TOP_RMS",
    "CUR-MTR_STD",
]
X_new = d[new_feature_name].values.astype(float)
X_new_train = X_new[train_idx]
X_new_test = X_new[test_idx]

new_mu = X_new_train.mean(axis=0)
new_sd = X_new_train.std(axis=0)

Z_new_train = (X_new_train - new_mu) / new_sd
Z_new_test = (X_new_test - new_mu) / new_sd

new_w, new_b = 학습(Z_new_train, y_train, lr=0.1, epochs=500)

new_tr_pred = 예측(Z_new_train, new_w, new_b)  # 학습용
new_te_pred = 예측(Z_new_test, new_w, new_b)  # 시험용
new_tr_r2 = R2(y_train, new_tr_pred)
new_te_r2 = R2(y_test, new_te_pred)
new_tr_mse = MSE(y_train, new_tr_pred)
new_te_mse = MSE(y_test, new_te_pred)

print("특징에 CUR-MTR_STD 를 하나 추가")
print(f"학습용 R2 {round(new_tr_r2, 4)} / 시험용 R2 {round(new_te_r2, 4)}")
# 학습용 R2 0.9984 / 시험용 R2 0.9982
print(f"학습용 MSE {round(new_tr_mse, 4)} / 시험용 MSE {round(new_te_mse, 4)}")
# 학습용 MSE 2.379 / 시험용 MSE 2.4411


# =====================================================================
# E. 함정 2 — 상관 순위와 실제 쓸모는 다르다
# =====================================================================
# [E1] 02 §6-1 처럼 특징을 하나씩 빼고 다시 학습해, 학습용·시험용 R2 를 각각 찍으세요.
#      (4개 특징이니 4줄이 나옵니다. 힌트: Z_train[:, 남길])
#
#      ★ 먼저 예상하고 적으세요. 실행은 그다음에. ★
#        A3 에서 본 상관을 보면 VIB-BOT_RMS 가 0.74 로 1등,
#        VIB-TOP_RMS 는 0.09 로 사실상 무관해 보입니다.
#        그럼 VIB-TOP_RMS 를 빼도 점수가 안 변하겠죠?
#      내 예상:
# TODO
print("E. 함정")
for 뺄 in range(4):
    남길 = [j for j in range(4) if j != 뺄]
    w_, b_ = 학습(Z_train[:, 남길], y_train)
    r_tr = R2(y_train, 예측(Z_train[:, 남길], w_, b_))
    r_te = R2(y_test, 예측(Z_test[:, 남길], w_, b_))
    print(f"{feature_name[뺄]:4s} 제외")
    print(f"학습용 R2 {round(r_tr, 4)}")
    print(f"시험용 R2 {round(r_te, 4)}")

# VIB-BOT_RMS 제외
# 학습용 R2 0.7748
# 시험용 R2 0.684
# VIB-BOT_PTP 제외
# 학습용 R2 0.7899
# 시험용 R2 0.5271
# VIB-BOT_KUR 제외
# 학습용 R2 0.7856
# 시험용 R2 0.5576
# VIB-TOP_RMS 제외
# 학습용 R2 0.6741
# 시험용 R2 0.3392


# [E2] 실행 결과를 보고 답하세요.
#        (1) 예상이 맞았나요? 예상이 틀림
#        (2) VIB-BOT_RMS 를 뺐을 때 시험용 점수가 어떻게 됐습니까?
#            왜 그럴까요?  힌트: d[특징이름].corr() 를 찍어 보세요.
#                                 VIB-BOT_RMS 와 VIB-BOT_PTP 의 상관은?
#            둘의 상관이 0.9534로 매우 높음. 서로 비슷한 정보를 많이 가지고 있어서 인 것 같음.
#        (3) VIB-TOP_RMS 를 뺐을 때는요? 정답과 상관이 0.09 밖에 안 되는데 왜?
#            단순 상관은 낮지만 다른 특징들과 함께 사용할 때 추가적인 정보를 제공함.
#            해당 특징을 빼자 R2가 약 0.3까지 크게 떨어짐
#        (4) 여기서 얻을 교훈을 한 줄로 적으세요.
#      내 답: 단순 상관 계수만 보고 특징의 쓸모를 판단하면 안 됨.
#            다른 특징들과 함께 사용했을 때의 실제 모델 성능도 확인해야 함.
# TODO
print(d[feature_name].corr())
#              VIB-BOT_RMS  VIB-BOT_PTP  VIB-BOT_KUR  VIB-TOP_RMS
# VIB-BOT_RMS     1.000000     0.953446    -0.348855     0.587099
# VIB-BOT_PTP     0.953446     1.000000    -0.190727     0.669398
# VIB-BOT_KUR    -0.348855    -0.190727     1.000000     0.066631
# VIB-TOP_RMS     0.587099     0.669398     0.066631     1.000000


# =====================================================================
# F. 함정 3 — 섞어서 자른 게 정말 옳았나
# =====================================================================
# [F1] 이 데이터는 MEAS_DT(측정시각) 순으로 정렬돼 있습니다. 1월부터 12월까지.
#      02 에서는 "섞고 잘라라, 안 섞으면 편향된다" 고 배웠죠.
#      이번엔 반대로 해 보세요 — 섞지 말고 앞 399행을 학습용, 뒤 171행을 시험용으로.
#      (즉 1~9월로 배워서 10~12월을 맞히기)
#
#      ★ 답하세요 ★
#        (1) 시험용 R2 가 섞었을 때(C3)보다 높나요, 낮나요?
#        (2) 결과가 예상과 다를 겁니다. 그래도 실무에서 예측 모델을 만들 때는
#            보통 이 '시간순 분할' 쪽을 씁니다. 왜 그럴까요?
#            힌트: 현장에 배포된 모델이 맞혀야 하는 데이터는 '언제' 것입니까?
#      내 답: (1) 랜덤 분할 했을 때의 R2보다 높게 나옴.
#             (2) 실제 현장에서는 과거 데이터로 학습한 모델이 이후 들어오는 미래 데이터를 예측해야 함.
#                 시간순 분할이 실제 사용 상황을 더 잘 반영함.
# TODO
print("F. 함정 3")
time_X_train = X[:399]
time_X_test = X[399:]

time_y_train = y[:399]
time_y_test = y[399:]

time_mu = time_X_train.mean(axis=0)
time_sd = time_X_test.mean(axis=0)

time_Z_train = (time_X_train - time_mu) / time_sd
time_Z_test = (time_X_test - time_mu) / time_sd

time_w, time_b = 학습(time_Z_train, time_y_train)

time_train_pred = 예측(time_Z_train, time_w, time_b)
time_test_pred = 예측(time_Z_test, time_w, time_b)

time_train_r2 = R2(time_y_train, time_train_pred)
time_test_r2 = R2(time_y_test, time_test_pred)

print(
    f"시간순 학습용 R2 {round(time_train_r2, 4)} / 시간순 시험용 R2 {round(time_test_r2, 4)}"
)
# 시간순 학습용 R2 0.7449 / 시간순 시험용 R2 0.777

# =====================================================================
# G. 데이터가 몇 대면 충분한가
# =====================================================================
# [G1] 학습용을 5 / 10 / 30 / 100 / 399 대로 바꿔 가며 학습하고,
#      학습용 R2 와 시험용 R2 를 표처럼 찍으세요. (적은 데이터는 epochs 를 늘리세요)
# _
#      ★ 답하세요 ★
#        (1) 시험용 R2 가 음수로 나오는 구간이 있습니다. 음수는 무슨 뜻입니까?
#            힌트: R2 = 0 이 '무조건 평균만 대답하는 모델' 입니다.
#        (2) 데이터가 늘 때 학습용 점수와 시험용 점수는 각각 어느 방향으로 움직입니까?
#        (3) 이 설비에서 쓸 만한 모델을 만들려면 최소 몇 건쯤 필요해 보입니까?
#      내 답:
#        (1) R2가 음수이면 평균값만 예측하는 모델보다도 성능이 안 좋다는 뜻.
#        (2) 데이터가 늘어날 수록 학습용 R2는 대체로 낮아지고 시험용 R2는 대체로 높아짐.
#            적은 데이터를 외우는 것 보다 일반적인 규칙을 배우게 된 것으로 볼 수 있음.
#        (3) 이 결과에서는 최소 100건 정도부터 쓸 만 하다고 판단할 수 있을 것 같음.
#            30건까지는 시험용 R2가 음수였지만 100건에서는 0.5873이 나옴.
# TODO
print("G. 데이터가 몇 대면 충분한가")
train_size = [5, 10, 30, 100, 399]

for size in train_size:
    w_, b_ = 학습(time_Z_train[:size], time_y_train[:size], epochs=2000)
    train_r2 = R2(time_y_train[:size], 예측(time_Z_train[:size], w_, b_))
    test_r2 = R2(time_y_test, 예측(time_Z_test, w_, b_))
    print(f"학습 데이터 {size}개")
    print(f"학습용 R2 {train_r2:.4f} / 시험용 R2 {test_r2:.4f}")
# 학습 데이터 5개
# 학습용 R2 0.9611 / 시험용 R2 -0.0198
# 학습 데이터 10개
# 학습용 R2 0.9153 / 시험용 R2 -2.4524
# 학습 데이터 30개
# 학습용 R2 0.8562 / 시험용 R2 -0.4951
# 학습 데이터 100개
# 학습용 R2 0.7739 / 시험용 R2 0.5873
# 학습 데이터 399개
# 학습용 R2 0.7452 / 시험용 R2 0.7750

# =====================================================================
# H. 마무리 — 보고서 3줄
# =====================================================================
# 팀장에게 보고한다고 치고, 아래 세 줄을 채우세요.
#
#   1) 전류계가 고장 났을 때 진동으로 전류를 추정할 수 있는가? 조건부로 가능
#      근거 점수: 시간순 분할 시험용 R2 0.7770
#
#   2) 이 모델을 쓸 때 반드시 붙여야 할 경고 문구 한 줄:
#      전류계에서 얻은 특징을 입력에 포함하면 데이터 누수가 발생하므로 사용하면 안 됨.
#
#   3) 점수를 더 올리려면 다음에 뭘 해 보겠는가? (한 가지만, 이유와 함께)
#      학습 데이터를 더 확보할 것.
#      데이터가 늘어날 수록 시험용 R2가 개선되는 경향을 보였기 때문.
#
# =====================================================================
