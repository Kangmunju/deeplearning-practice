# =====================================================================
#  문제 8 — 신경망 하나로 회귀·이진·다중을 다  (11 · 12)
# =====================================================================
#  [목표]
#    수업에서는 회귀용(09), 이진용(11), 다중용(12 B부), 깊은 망(12 A부) 코드를 '따로' 짰습니다.
#    오늘은 그걸 '하나' 로 합칩니다. 합치다 보면 "진짜 바뀌는 곳이 어디인가" 가 보입니다.
#
#  [만들어야 하는 것]
#    - 구조를 리스트로 받기
#    - 은닉 활성화: sigmoid / relu
#    - 출력 종류: 회귀 / 이진 / 다중
#    - 초기화: he / 작게 / zero
#    - seed 고정, 학습 중 손실 기록
#
#  ★ 조건 ★  역전파 함수는 하나만 사용
# =====================================================================

import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "데이터")


# =====================================================================
# 공통 신경망
# =====================================================================


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def 초기화(구조, 방식="he", seed=0):
    rng = np.random.RandomState(seed)
    W = []
    b = []

    for i in range(len(구조) - 1):
        입력수 = 구조[i]
        출력수 = 구조[i + 1]

        if 방식 == "he":
            wi = rng.randn(입력수, 출력수) * np.sqrt(2 / 입력수)
        elif 방식 == "작게":
            wi = rng.randn(입력수, 출력수) * 0.5
        else:
            wi = np.zeros((입력수, 출력수))

        W.append(wi)
        b.append(np.zeros((1, 출력수)))

    return W, b


def 순전파(X, W, b, 활성화="relu", 출력종류="회귀"):
    A = [X]
    Z = []

    for i in range(len(W)):
        z = A[-1] @ W[i] + b[i]
        Z.append(z)

        if i == len(W) - 1:
            if 출력종류 == "이진":
                a = sigmoid(z)
            elif 출력종류 == "다중":
                a = softmax(z)
            else:
                a = z
        else:
            if 활성화 == "sigmoid":
                a = sigmoid(z)
            else:
                a = np.maximum(0, z)

        A.append(a)

    return A, Z


def 손실(y, 예측, 출력종류):
    p = np.clip(예측, 1e-12, 1 - 1e-12)

    if 출력종류 == "회귀":
        return np.mean((예측 - y) ** 2)

    if 출력종류 == "이진":
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return -np.mean(np.sum(y * np.log(p), axis=1))


def 역전파(y, W, A, Z, 활성화="relu", 출력종류="회귀"):
    n = len(y)
    dW = []
    db = []

    for wi, bi in zip(W, Z):
        dW.append(np.zeros_like(wi))
        db.append(np.zeros((1, bi.shape[1])))

    # 출력층: 출력 종류에 따라 달라지는 부분
    if 출력종류 == "회귀":
        delta = 2 * (A[-1] - y) / y.size
    else:
        delta = (A[-1] - y) / n

    # 나머지 역전파는 세 문제에서 동일
    for i in range(len(W) - 1, -1, -1):
        dW[i] = A[i].T @ delta
        db[i] = np.sum(delta, axis=0, keepdims=True)

        if i > 0:
            delta = delta @ W[i].T

            if 활성화 == "sigmoid":
                delta = delta * A[i] * (1 - A[i])
            else:
                delta = delta * (Z[i - 1] > 0)

    return dW, db


def 학습(
    X,
    y,
    구조,
    활성화="relu",
    출력종류="회귀",
    초기화방식="he",
    lr=0.05,
    epochs=3000,
    seed=0,
):

    W, b = 초기화(구조, 초기화방식, seed)
    기록 = []

    for epoch in range(epochs):
        A, Z = 순전파(X, W, b, 활성화, 출력종류)
        loss = 손실(y, A[-1], 출력종류)
        기록.append(loss)

        dW, db = 역전파(y, W, A, Z, 활성화, 출력종류)

        for i in range(len(W)):
            W[i] -= lr * dW[i]
            b[i] -= lr * db[i]

    return W, b, 기록


def 예측(X, W, b, 활성화, 출력종류):
    A, Z = 순전파(X, W, b, 활성화, 출력종류)
    return A[-1]


def 정확도(X, y, W, b, 활성화="relu"):
    p = 예측(X, W, b, 활성화, "다중")
    판정 = np.argmax(p, axis=1)
    정답 = np.argmax(y, axis=1)
    return np.mean(판정 == 정답)


# ---------------------------------------------------------------------
# [8-1] 기울기 검사
#   샘플 6개, 입력 3개, 구조 [3, 5, 4, K]
#   (회귀 / 이진 / 다중) × (sigmoid / relu)
#   모든 W·b 를 수치미분과 대조
#   확인: 최대 차이 1e-7 미만
#
#   ★ 설명 ★ 출력 종류에 따라 역전파에서 바뀐 줄은 몇 줄인가요?
# ---------------------------------------------------------------------

print("[8-1] 기울기 검사")

rng = np.random.RandomState(0)
작은X = rng.randn(6, 3)

for 출력종류 in ["회귀", "이진", "다중"]:
    for 활성화 in ["sigmoid", "relu"]:
        if 출력종류 == "다중":
            K = 3
            정답번호 = rng.randint(0, K, 6)
            작은y = np.eye(K)[정답번호]
        elif 출력종류 == "이진":
            K = 1
            작은y = rng.randint(0, 2, (6, 1)).astype(float)
        else:
            K = 1
            작은y = rng.randn(6, 1)

        W, b = 초기화([3, 5, 4, K], "작게", 0)

        A, Z = 순전파(작은X, W, b, 활성화, 출력종류)
        dW, db = 역전파(작은y, W, A, Z, 활성화, 출력종류)

        h = 1e-5
        최대차이 = 0.0

        # 모든 가중치 수치미분
        for layer in range(len(W)):
            for i in range(W[layer].shape[0]):
                for j in range(W[layer].shape[1]):
                    원래값 = W[layer][i, j]

                    W[layer][i, j] = 원래값 + h
                    p1 = 예측(작은X, W, b, 활성화, 출력종류)
                    L1 = 손실(작은y, p1, 출력종류)

                    W[layer][i, j] = 원래값 - h
                    p2 = 예측(작은X, W, b, 활성화, 출력종류)
                    L2 = 손실(작은y, p2, 출력종류)

                    W[layer][i, j] = 원래값

                    수치기울기 = (L1 - L2) / (2 * h)
                    차이 = abs(수치기울기 - dW[layer][i, j])
                    최대차이 = max(최대차이, 차이)

            # 모든 편향 수치미분
            for j in range(b[layer].shape[1]):
                원래값 = b[layer][0, j]

                b[layer][0, j] = 원래값 + h
                p1 = 예측(작은X, W, b, 활성화, 출력종류)
                L1 = 손실(작은y, p1, 출력종류)

                b[layer][0, j] = 원래값 - h
                p2 = 예측(작은X, W, b, 활성화, 출력종류)
                L2 = 손실(작은y, p2, 출력종류)

                b[layer][0, j] = 원래값

                수치기울기 = (L1 - L2) / (2 * h)
                차이 = abs(수치기울기 - db[layer][0, j])
                최대차이 = max(최대차이, 차이)

        print(f"{출력종류:4s} / {활성화:7s} 최대 차이 {최대차이:.2e}")

        assert 최대차이 < 1e-7

# ★ 설명 ★
# 출력 종류에 따라 달라지는 핵심은 출력층 delta 계산 부분
# 회귀는 MSE 미분, 이진은 sigmoid+BCE, 다중은 softmax+교차엔트로피
# 이진과 다중은 둘 다 (예측-정답)/n으로 정리됨
# 이후 은닉층으로 기울기를 전달하는 사슬법칙은 완전히 동일


# ---------------------------------------------------------------------
# [8-2] U자 데이터 — 깊이의 대가
#   (a) [1,8,1], sigmoid, 작게, lr 0.05, 3000 epochs
#   (b) 은닉 6층을 seed 0~4로 비교
#
#   ★ 먼저 예측 ★ ①②③ 중 잘 되는 것과 안 되는 것은?
# ---------------------------------------------------------------------

print("\n[8-2] U자 데이터")

온도 = np.array([10, 15, 20, 25, 30, 35, 40], dtype=float)
불량 = np.array([9, 5, 2, 1, 2, 5, 9], dtype=float).reshape(-1, 1)

U_X = ((온도 - 온도.mean()) / 온도.std()).reshape(-1, 1)

W_u, b_u, 기록_u = 학습(
    U_X,
    불량,
    [1, 8, 1],
    활성화="sigmoid",
    출력종류="회귀",
    초기화방식="작게",
    lr=0.05,
    epochs=3000,
    seed=0,
)

print(f"얕은 망 마지막 손실 {기록_u[-1]:.6f}")
print(
    f"얕은 망 학습 후 손실 {손실(불량, 예측(U_X, W_u, b_u, 'sigmoid', '회귀'), '회귀'):.6f}"
)

# ★ 먼저 예측 ★
# ① sigmoid + 작게: 깊어질수록 기울기 소실로 학습 실패 예상
# ② ReLU + He + 작은 lr: 상대적으로 안정적으로 학습 예상
# ③ ReLU + He + 큰 lr: 빠르지만 일부 seed에서 죽은 ReLU 발생 가능

깊은구조 = [1, 8, 8, 8, 8, 8, 8, 1]

설정들 = [
    ("① sigmoid + 작게", "sigmoid", "작게", 0.05),
    ("② relu + he 0.01", "relu", "he", 0.01),
    ("③ relu + he 0.05", "relu", "he", 0.05),
]

print(f"{'설정':<22} {'seed':>6} {'최종 손실':>14}")

for 이름, 활성화, 초기화방식, lr in 설정들:
    for seed in range(5):
        W_u, b_u, 기록_u = 학습(
            U_X,
            불량,
            깊은구조,
            활성화=활성화,
            출력종류="회귀",
            초기화방식=초기화방식,
            lr=lr,
            epochs=3000,
            seed=seed,
        )

        최종예측 = 예측(U_X, W_u, b_u, 활성화, "회귀")
        최종손실 = 손실(불량, 최종예측, "회귀")

        print(f"{이름:<22} {seed:>6} {최종손실:>14.6f}")

# 실패한 망이 평균만 출력하면 손실은 정답의 분산과 같아짐
평균 = 불량.mean()
분산 = 불량.var()

직선계수 = np.polyfit(U_X.flatten(), 불량.flatten(), 1)
직선예측 = np.polyval(직선계수, U_X.flatten())
직선손실 = np.mean((직선예측 - 불량.flatten()) ** 2)

print(f"\n불량 평균 {평균:.6f}")
print(f"불량 분산 {분산:.6f}")
print(f"최적 직선 손실 {직선손실:.6f}")

# 불량 평균 = 4.714286
# 불량 분산 = 9.346939
# U자 데이터는 좌우 대칭이므로 최적 직선의 기울기는 거의 0
# 따라서 최적 직선도 평균 4.714286만 예측하고 MSE는 9.346939
# 이 값에서 멈춘 망은 입력과 관계없이 평균만 출력하는 망이 된 것
# ① sigmoid는 깊은 층에서 작은 미분값이 계속 곱해져 기울기 소실 발생

# ③ seed 0의 죽은 뉴런 확인
print("\n③ seed 0의 은닉층 확인")

W_bad, b_bad, 기록_bad = 학습(
    U_X,
    불량,
    깊은구조,
    활성화="relu",
    출력종류="회귀",
    초기화방식="he",
    lr=0.05,
    epochs=3000,
    seed=0,
)

A_bad, Z_bad = 순전파(U_X, W_bad, b_bad, "relu", "회귀")

for i in range(1, len(A_bad) - 1):
    죽은뉴런 = np.sum(np.all(A_bad[i] == 0, axis=0))
    print(f"은닉층 {i}: 모든 샘플에서 출력 0인 뉴런 {죽은뉴런}개")

print(f"출력 7개 {A_bad[-1].flatten().round(4)}")

# ②와 ③은 초기화가 같아도 학습률이 다름
# 큰 학습률은 가중치를 한 번에 크게 바꿔 ReLU 입력을 음수 구간으로 밀어 넣을 수 있음
# 음수 구간에서 기울기가 0이므로 뉴런이 죽으면 학습이 정체될 수 있음


# ---------------------------------------------------------------------
# [8-3] XOR — 이진
#   [2, 은닉수, 1], sigmoid, 작게, lr 1.0, 5000 epochs
#   은닉수 1 / 2 / 4, seed 0~4
#
#   ★ 먼저 예측 ★ 은닉 1개면 최대 몇 개?
# ---------------------------------------------------------------------

print("\n[8-3] XOR")

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

y_xor = np.array([[0], [1], [1], [0]], dtype=float)

# ★ 먼저 예측 ★
# 은닉 1개는 XOR의 두 대각선 영역을 충분히 구분하기 어려움
# 단일 퍼셉트론과 마찬가지로 4/4 학습은 불가능할 것으로 예상

print(f"{'은닉수':<10} {'seed':>8} {'맞힌 수':>10} {'손실':>12}")

for 은닉수 in [1, 2, 4]:
    for seed in range(5):
        W_xor, b_xor, 기록_xor = 학습(
            X_xor,
            y_xor,
            [2, 은닉수, 1],
            활성화="sigmoid",
            출력종류="이진",
            초기화방식="작게",
            lr=1.0,
            epochs=5000,
            seed=seed,
        )

        p = 예측(X_xor, W_xor, b_xor, "sigmoid", "이진")
        판정 = (p >= 0.5).astype(int)
        맞힌수 = np.sum(판정 == y_xor)

        print(f"{은닉수:<10} {seed:>8} {맞힌수:>10}/4 {기록_xor[-1]:>12.6f}")

# 은닉 2개면 이론상 XOR 표현이 가능하지만 모든 초기값에서 학습이 성공하지는 않음
# 경사하강은 초기값과 손실함수의 형태에 영향을 받음
# 은닉 4개처럼 여유 있는 구조가 실제 학습에서는 더 안정적일 수 있음
# 이론적으로 표현 가능한 최소 크기와 실제 학습이 잘 되는 크기는 다름


# ---------------------------------------------------------------------
# [8-4] ai4i — 고장이 났다면, 무슨 고장인가 (4 분류)
#   네 고장 종류 중 정확히 하나만 1인 행만 사용
#   확인: 307행 (43 / 106 / 80 / 78)
#   train 182 / val 62 / test 63
# ---------------------------------------------------------------------

print("\n[8-4] ai4i 고장 종류 분류")

df = pd.read_csv(os.path.join(DATA, "11_설비센서_ai4i_전체.csv"), encoding="utf-8-sig")

센서 = ["공기온도", "공정온도", "회전수", "토크", "공구마모"]
고장종류 = ["공구마모고장", "방열고장", "전력고장", "과부하고장"]

# 정확히 하나의 고장 종류만 1인 행 선택
조건 = df[고장종류].sum(axis=1) == 1
고장df = df.loc[조건].copy()

X4 = 고장df[센서].values.astype(float)
y4 = np.argmax(고장df[고장종류].values, axis=1)

print(f"전체 {len(y4)}행")
print(f"클래스별 개수 {np.bincount(y4, minlength=4)}")

# 클래스 0 → 1 → 2 → 3 순서로 같은 rng 사용
rng = np.random.RandomState(0)

train_idx = []
val_idx = []
test_idx = []

for k in range(4):
    idx = rng.permutation(np.where(y4 == k)[0])

    a = int(len(idx) * 0.6)
    b = int(len(idx) * 0.8)

    train_idx.extend(idx[:a])
    val_idx.extend(idx[a:b])
    test_idx.extend(idx[b:])

X_train = X4[train_idx]
X_val = X4[val_idx]
X_test = X4[test_idx]

y_train = np.eye(4)[y4[train_idx]]
y_val = np.eye(4)[y4[val_idx]]
y_test = np.eye(4)[y4[test_idx]]

print(f"train {len(y_train)} / val {len(y_val)} / test {len(y_test)}")

# train 기준 표준화
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)

Z_train = (X_train - mu) / sd
Z_val = (X_val - mu) / sd
Z_test = (X_test - mu) / sd


# ---------------------------------------------------------------------
# [8-4] (a) 초기화에 따른 첫 손실
#   [5,16,4], relu, zero / 작게 / he
# ---------------------------------------------------------------------

# ★ 먼저 예측 ★
# zero 초기화면 모든 로짓이 0
# softmax 확률은 각 클래스 1/4
# 교차엔트로피 = -log(1/4) = log(4) ≈ 1.386294

print("\n[8-4a] 초기화별 첫 손실")

for 방식 in ["zero", "작게", "he"]:
    W0, b0 = 초기화([5, 16, 4], 방식, 0)
    p0 = 예측(Z_train, W0, b0, "relu", "다중")
    loss0 = 손실(y_train, p0, "다중")

    print(f"{방식:<6} 첫 손실 {loss0:.6f}")

# zero는 모든 클래스 확률이 0.25이므로 손실이 정확히 log(4)
# 나머지는 난수 가중치로 클래스 확률이 달라져 손실도 달라짐
# 실제 정답과 맞지 않는 확신이 생기면 log(4)보다 손실이 커질 수 있음


# ---------------------------------------------------------------------
# [8-4] (b) 은닉수와 학습률 비교
#   은닉 4 / 16 / 64 × lr 0.05 / 0.3
#   relu + he, 3000 epochs
#   val로 모델 선택, 같으면 작은 망
# ---------------------------------------------------------------------

print("\n[8-4b] 모델 비교")
print(f"{'은닉수':>8} {'lr':>8} {'train 정확도':>14} {'val 정확도':>14}")

최고val = -1
최적은닉 = None
최적lr = None
최적W = None
최적b = None

for 은닉수 in [4, 16, 64]:
    for lr in [0.05, 0.3]:
        W4, b4, 기록4 = 학습(
            Z_train,
            y_train,
            [5, 은닉수, 4],
            활성화="relu",
            출력종류="다중",
            초기화방식="he",
            lr=lr,
            epochs=3000,
            seed=0,
        )

        train_acc = 정확도(Z_train, y_train, W4, b4)
        val_acc = 정확도(Z_val, y_val, W4, b4)

        print(f"{은닉수:>8} {lr:>8.2f} {train_acc:>14.4f} {val_acc:>14.4f}")

        # 같은 val 정확도면 먼저 등장한 작은 은닉수 선택
        if val_acc > 최고val:
            최고val = val_acc
            최적은닉 = 은닉수
            최적lr = lr
            최적W = W4
            최적b = b4

print(f"\n선택된 은닉수 {최적은닉}")
print(f"선택된 학습률 {최적lr}")
print(f"최고 val 정확도 {최고val:.4f}")

# 모델 선택을 마친 뒤 test를 처음 평가
test_acc = 정확도(Z_test, y_test, 최적W, 최적b)
print(f"최종 test 정확도 {test_acc:.4f}")

# train 정확도로 선택하면 학습 데이터를 가장 잘 외운 큰 망을 고르기 쉬움
# 하지만 train 정확도는 새로운 데이터에 대한 일반화 성능을 보장하지 않음
# 따라서 val로 설정을 선택하고 test는 마지막 평가에만 사용


# ---------------------------------------------------------------------
# [8-4] (c) 혼동행렬
#   행 = 실제, 열 = 예측
# ---------------------------------------------------------------------

print("\n[8-4c] 혼동행렬")

p_test = 예측(Z_test, 최적W, 최적b, "relu", "다중")
실제 = np.argmax(y_test, axis=1)
판정 = np.argmax(p_test, axis=1)

혼동행렬 = np.zeros((4, 4), dtype=int)

for a, p in zip(실제, 판정):
    혼동행렬[a, p] += 1

print(pd.DataFrame(혼동행렬, index=고장종류, columns=고장종류))

오답행렬 = 혼동행렬.copy()
np.fill_diagonal(오답행렬, 0)

print("\n예측 종류별 오답 수")
for i in range(4):
    print(f"{고장종류[i]}: {오답행렬[:, i].sum()}개")

print(f"총 오답 수 {오답행렬.sum()}개")

# 오답이 몰린 열은 위 표에서 직접 확인
# 전력고장은 토크와 회전수의 곱, 방열고장은 온도차와 회전수와 관련
# 센서 조건이 겹치는 경우 다른 고장 종류로 혼동할 수 있음
# test가 63개이므로 1개 차이는 약 1.59%p
# 정확도 0.95와 0.97은 약 1.26개 차이로, 실제로는 1~2개 차이에 불과


# ---------------------------------------------------------------------
# [8-4] (d) zero 초기화의 대가
#   선택된 구조와 lr로 zero 초기화 학습
#   val 정확도와 은닉층 W 열 비교
# ---------------------------------------------------------------------

print("\n[8-4d] zero 초기화")

W_zero, b_zero, 기록_zero = 학습(
    Z_train,
    y_train,
    [5, 최적은닉, 4],
    활성화="relu",
    출력종류="다중",
    초기화방식="zero",
    lr=최적lr,
    epochs=3000,
    seed=0,
)

zero_val = 정확도(Z_val, y_val, W_zero, b_zero)

print(f"zero 초기화 val 정확도 {zero_val:.4f}")

# 첫 은닉층의 모든 뉴런 가중치가 같은지 확인
첫열 = W_zero[0][:, 0]
모두같음 = True

for j in range(1, W_zero[0].shape[1]):
    if not np.allclose(첫열, W_zero[0][:, j]):
        모두같음 = False

print(f"첫 은닉층 W의 열이 모두 같은가? {모두같음}")

val비율 = np.bincount(np.argmax(y_val, axis=1), minlength=4) / len(y_val)

print(f"val 클래스 비율 {val비율.round(4)}")
print(f"val 다수 클래스 비율 {val비율.max():.4f}")

# zero 초기화면 모든 은닉 뉴런이 같은 값에서 출발
# ReLU 입력도 0이므로 첫 은닉층으로 전달되는 기울기가 0
# 따라서 은닉층이 특징을 학습하지 못하고 출력 편향만 학습
# 결과적으로 가장 빈번한 클래스만 예측하는 모델이 됨
# val 정확도 약 0.34는 다수 클래스 비율과 비슷함
