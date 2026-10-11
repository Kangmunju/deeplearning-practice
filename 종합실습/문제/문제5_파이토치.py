# =====================================================================
#  문제 5 — 문제 2 를 파이토치로. 같은 답이 나와야 한다  (05)
# =====================================================================
#  [목표]  파이토치가 '별개의 마법'이 아니라 문제 2 의 그 루프라는 걸 숫자로 증명합니다.
#  [파일]  데이터/11_설비센서_ai4i_전체.csv
#  [공통]  문제 2 [2-4] 와 같은 입력·정답·분할·표준화 (회전수, 공기온도, 공정온도, 공구마모 → 토크)
#          텐서는 float32.  ★ 정답 텐서의 모양을 꼭 확인하세요 ★
# =====================================================================

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "데이터")

df = pd.read_csv(os.path.join(DATA, "11_설비센서_ai4i_전체.csv"), encoding="utf-8-sig")

특징 = ["회전수", "공기온도", "공정온도", "공구마모"]

X = df[특징].values.astype(float)
y = df["토크"].values.astype(float)

순서 = np.random.RandomState(0).permutation(len(X))

train_idx = 순서[:7000]
test_idx = 순서[7000:]

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

# 문제 2와 동일하게 train 기준으로 표준화
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)

Z_train = (X_train - mu) / sd
Z_test = (X_test - mu) / sd

# PyTorch 텐서는 float32, 정답은 (n, 1) 형태
Xt = torch.tensor(Z_train, dtype=torch.float32)
Xv = torch.tensor(Z_test, dtype=torch.float32)

yt = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
yv = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

print(f"입력 텐서 모양 {Xt.shape}")
print(f"정답 텐서 모양 {yt.shape}")


def R2(y, yhat):
    return 1 - torch.sum((y - yhat) ** 2) / torch.sum((y - y.mean()) ** 2)


# ---------------------------------------------------------------------
# [5-1] autograd 를 손 공식과 대조
#   w (4×1), b 를 0 으로 만들고 requires_grad 를 켜서, train 전체의 MSE 를 구한 뒤 backward 하세요.
#   같은 자리의 기울기를 '손 공식'(문제 2 에서 쓴 그것)으로도 계산해서 나란히 찍으세요.
#   확인값: db ≈ -80.01,  dw ≈ [17.51, 0.26, 0.16, -0.07]
#   ★ 설명 ★  db 가 왜 정확히 '-2 × 토크 평균' 인가요?
# ---------------------------------------------------------------------
print("\n[5-1] autograd와 손 공식 비교")

w = torch.zeros((4, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

예측 = Xt @ w + b
loss = torch.mean((예측 - yt) ** 2)

loss.backward()

# MSE = mean((예측 - 정답)^2)
# dw = 2 * X.T @ (예측 - 정답) / n
# db = 2 * mean(예측 - 정답)
오차 = 예측.detach() - yt

손_dw = 2 * Xt.T @ 오차 / len(yt)
손_db = 2 * 오차.mean()

print(f"autograd dw {w.grad.flatten().detach().numpy().round(4)}")
print(f"손 공식 dw {손_dw.flatten().numpy().round(4)}")
print(f"autograd db {b.grad.item():.4f}")
print(f"손 공식 db {손_db.item():.4f}")

# w와 b가 모두 0이므로 첫 예측값은 전부 0
# db = 2 * mean(0 - y) = -2 * 토크 평균
# 따라서 초기 절편 기울기는 약 -80.01
# autograd와 손 공식의 결과가 같으므로 자동미분도 같은 미분 계산을 수행


# ---------------------------------------------------------------------
# [5-2] 표준 루프
#   nn.Linear(4, 1) + SGD(lr 0.1), 전체 배치로 500 epochs. (torch.manual_seed(0))
#   확인값: 가중치·절편·train/test R² 가 문제 2 [2-4] 와 소수 셋째 자리까지 같아야 합니다.
#   ★ 설명 ★  nn.Linear 는 0 이 아니라 난수에서 출발합니다. 그런데도 같은 답이 나오는 이유는?
#   ★ 함정 체험 ★  정답 텐서를 (n,) 모양으로 두고 다시 돌려 보세요. 에러가 나나요? 결과는?
#                 (모델 출력은 (n,1) 입니다. 두 모양을 빼면 무슨 모양이 되나요?)
# ---------------------------------------------------------------------
print("\n[5-2] nn.Linear + SGD")

torch.manual_seed(0)

model = nn.Linear(4, 1)
criterion = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(500):
    opt.zero_grad()

    pred = model(Xt)
    loss = criterion(pred, yt)

    loss.backward()
    opt.step()

with torch.no_grad():
    train_pred = model(Xt)
    test_pred = model(Xv)

print(f"가중치 {model.weight.detach().numpy().flatten().round(4)}")
print(f"절편 {model.bias.item():.4f}")
print(f"train R² {R2(yt, train_pred).item():.4f}")
print(f"test R² {R2(yv, test_pred).item():.4f}")

# nn.Linear는 난수로 초기화되지만 선형회귀의 MSE는 볼록한 손실함수
# 적절한 학습률로 충분히 학습하면 초기값이 달라도 같은 최소점에 수렴
# 따라서 문제 2에서 가중치를 0으로 시작한 결과와 거의 같음

# ★ 함정 체험 ★
print("\n정답 텐서 모양 함정")

# 전체 7000개를 사용하면 (7000, 7000)으로 커지므로 5개만 사용
작은X = Xt[:5]
작은y = yt[:5]

with torch.no_grad():
    작은예측 = model(작은X)

잘못된y = 작은y.reshape(-1)

print(f"모델 출력 모양 {작은예측.shape}")
print(f"올바른 정답 모양 {작은y.shape}")
print(f"잘못된 정답 모양 {잘못된y.shape}")

잘못된오차 = 작은예측 - 잘못된y

print(f"잘못 계산된 오차 모양 {잘못된오차.shape}")
print(f"올바른 MSE {criterion(작은예측, 작은y).item():.4f}")
print(f"잘못된 MSE {torch.mean(잘못된오차**2).item():.4f}")

# (5,1) - (5,)는 브로드캐스팅되어 (5,5)가 됨
# 에러 없이 실행되지만 각 예측값이 모든 정답과 비교되어 잘못된 손실 계산
# 전체 7000개라면 (7000,7000)으로 확장되어 메모리도 크게 낭비
# 따라서 회귀 정답 텐서는 반드시 모델 출력과 같은 (n,1)로 맞춰야 함


# ---------------------------------------------------------------------
# [5-3] 미니배치
#   DataLoader(batch_size=64, shuffle=True) + Adam(lr 0.05), 20 epochs. (torch.manual_seed(0))
#   ★ 먼저 예측 ★  한 epoch 에 몇 걸음을 걷나요? 20 epochs 면 총 몇 걸음?
#   확인값: test R² ≈ 0.765
#   ★ 설명 ★  미니배치에서 손실이 매 걸음 매끈하게 줄지 않고 출렁이는 이유는?
# ---------------------------------------------------------------------

# ★ 먼저 예측 ★
# train 7000개를 64개씩 나누면 109개 배치와 마지막 24개 배치
# 한 epoch에 ceil(7000/64) = 110걸음
# 20 epochs에서는 총 110 * 20 = 2200걸음

print("\n[5-3] DataLoader + Adam")

torch.manual_seed(0)

dataset = TensorDataset(Xt, yt)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model2 = nn.Linear(4, 1)
opt2 = torch.optim.Adam(model2.parameters(), lr=0.05)

print(f"한 epoch 걸음 수 {len(loader)}")
print(f"20 epochs 총 걸음 수 {len(loader) * 20}")

for epoch in range(20):
    for xb, yb in loader:
        opt2.zero_grad()

        pred = model2(xb)
        loss = criterion(pred, yb)

        loss.backward()
        opt2.step()

    with torch.no_grad():
        epoch_loss = criterion(model2(Xt), yt)

    print(f"epoch {epoch + 1:2d} / 20  train MSE {epoch_loss.item():.4f}")

with torch.no_grad():
    test_pred2 = model2(Xv)

print(f"test R² {R2(yv, test_pred2).item():.4f}")

# 미니배치는 매 걸음 서로 다른 64개 데이터로 기울기를 계산
# 배치마다 데이터의 특성이 다르므로 기울기 방향과 손실이 조금씩 달라짐
# 따라서 손실이 매번 매끄럽게 감소하지 않고 출렁일 수 있음
# 전체 데이터 기준으로는 대체로 최적값에 가까워지는 방향으로 학습


# =====================================================================
# [5-4] 생각해 볼 것
# =====================================================================
# Q1 opt.zero_grad() 를 빼면 어떻게 되나요? 직접 빼고 돌려 보세요.
# Q2 이 문제에 파이토치를 쓰는 게 맞는 선택인가요? 언제 파이토치를 꺼내야 하나요?
# =====================================================================
print("\n[5-4] 생각해 볼 것")

# Q1. zero_grad()를 빼면 기울기가 초기화되지 않고 누적됨
#     이전 배치의 기울기까지 계속 더해져 의도한 경사하강이 이루어지지 않음

# 실제로 첫 두 걸음의 기울기 누적 확인
torch.manual_seed(0)

실험모델 = nn.Linear(4, 1)
실험opt = torch.optim.SGD(실험모델.parameters(), lr=0.1)

for step in range(2):
    # 실험opt.zero_grad() 생략
    실험loss = criterion(실험모델(Xt), yt)
    실험loss.backward()

    print(f"zero_grad 생략 {step + 1}걸음 기울기 {실험모델.bias.grad.item():.4f}")

    실험opt.step()

# 첫 걸음의 기울기가 다음 걸음에도 남아 누적되는 것을 확인
# 실제 학습에서는 매 걸음 opt.zero_grad()로 초기화해야 함

# Q2. 입력 4개로 토크를 예측하는 단순 선형회귀에는 NumPy나 sklearn으로도 충분
#     PyTorch는 신경망처럼 층이 많고 복잡한 모델을 학습할 때 유리
#     자동미분, GPU 연산, 미니배치 학습이 필요한 경우에 사용하는 것이 적절
