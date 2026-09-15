# =====================================================================
#  05. 파이토치 — 01~03 의 '걷기'를 그대로, 미분만 자동으로. 그리고 언제 쓰나
#  실행: python 05_파이토치_같은일을_루프로.py   (수업코드 폴더에서)
# =====================================================================

import os
import numpy as np
import pandas as pd
import torch                     # 파이토치. 이름이 torch 인 건 역사적 이유 (Torch 라는 옛 도구의 파이썬판)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "수업용데이터")   # data/수업용데이터/
df = pd.read_csv(os.path.join(DATA, "11_설비센서_ai4i.csv"), encoding="utf-8-sig")
특징이름 = ["공기온도", "회전수", "토크", "공구마모"]
torch.manual_seed(0)             # 파이토치는 w, b 시작값을 난수로 잡습니다 (01 은 0 에서 출발). 고정해야 매번 같은 결과


# =====================================================================
# 1. 텐서 — numpy 배열의 파이토치 판 (딱 하나만 조심: float32)
# =====================================================================
x = df["공기온도"].values
y = df["공정온도"].values
x_t = torch.tensor(x, dtype=torch.float32)       # numpy → 텐서. dtype=torch.float32 를 습관처럼 붙입니다
y_t = torch.tensor(y, dtype=torch.float32)       # 왜? 파이토치 부품들은 전부 float32 를 기대. 정수(int)나 float64 가 섞이면 에러
print("[1] 텐서:", x_t[:3], "/ 모양", x_t.shape, "/ 타입", x_t.dtype)
print("    다시 numpy 로:", x_t[:3].numpy())      # .numpy() 로 언제든 되돌립니다. 둘은 자유롭게 오갑니다


# =====================================================================
# 2. 자동 미분 — 01 의 '밟아보기'를 파이토치가 대신
# =====================================================================
m, s = x_t.mean(), x_t.std(unbiased=False)       # 표준화 (unbiased=False = numpy 와 같은 방식의 표준편차. 안 붙이면 살짝 다른 값)
z_t = (x_t - m) / s

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
손실 = ((y_t - (w * z_t + b)) ** 2).mean()        # 01 의 MSE 그대로. 이 계산이 '기록'됩니다
손실.backward()                                  # ← 미분! 이 한 줄이 01 의 기울기_밟아보기(). 결과는 w.grad, b.grad 에 담김
print("\n[2] autograd 가 구한 기울기: w 방향", round(w.grad.item(), 3), "/ b 방향", round(b.grad.item(), 3))

h = 1e-4
z64, y64 = z_t.numpy().astype(np.float64), y_t.numpy().astype(np.float64)
L = lambda wv, bv: np.mean((y64 - (wv * z64 + bv)) ** 2)
print("    밟아보기로 잰 기울기:      w 방향", round((L(h, 0) - L(-h, 0)) / (2 * h), 3),

      "/ b 방향", round((L(0, h) - L(0, -h)) / (2 * h), 3), "← 같다")


# =====================================================================
# 3. 파이토치 표준 루프 — 부품 세 개 + 네 줄
# =====================================================================
Z = z_t.reshape(-1, 1)          # nn.Linear 는 (설비 수, 센서 수) 세로 표를 받습니다 (사이킷런 규칙 1 과 같음)
Y = y_t.reshape(-1, 1)

torch.manual_seed(0)
model = torch.nn.Linear(1, 1)                              # 입력 1개(공기온도) → 출력 1개(공정온도). w 1개 + b 1개
loss_fn = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)          # model.parameters() = 이 모델의 손잡이들(w, b). "얘네를 갱신해라"

print("\n[3] 학습 루프 — 이 네 줄이 01 의 '한 걸음'")
for epoch in range(300):                # 01 과 같은 300 에폭
    opt.zero_grad()                     # ① 기울기 비우기 (누적 방지. 2번의 주의)
    loss = loss_fn(model(Z), Y)         # ② 예측 → 손실          (model(Z) = w·z + b 를 200대에 대해)
    loss.backward()                     # ③ 기울기 자동 계산       (01 의 밟아보기)
    opt.step()                          # ④ 한 걸음               (01 의 w = w − lr·기울기)
    if epoch in (0, 10, 30, 100, 299):
        print(f"    epoch {epoch:3d}  손실 {loss.item():.4f}")

w_z, b_z = model.weight.item(), model.bias.item()          # 학습된 손잡이. 사이킷런의 coef_, intercept_ 에 해당
print(f"    원래 눈금: 공정온도 ≈ {w_z / s.item():.4f} × 공기온도 + {b_z - w_z * m.item() / s.item():.4f}")
print("    01 (손) / 04 (사이킷런): 0.9840 × 공기온도 + 14.7799  ← 같다 (끝자리 0.0002 차이는 float32 탓)")

# =====================================================================
# 4. 다변수 + train/test — 02 를 파이토치로 (나누기·표준화는 사이킷런을 빌려 씀)
# =====================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[특징이름].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   # 04 2번와 같은 분할
scaler = StandardScaler().fit(X_train)
to_t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)      # numpy → float32 텐서 도우미. 매번 길게 안 쓰려고
Ztr, Zte = to_t(scaler.transform(X_train)), to_t(scaler.transform(X_test))
Ytr, Yte = to_t(y_train).reshape(-1, 1), to_t(y_test).reshape(-1, 1)

torch.manual_seed(0)
model4 = torch.nn.Linear(4, 1)                             # 입력 4개(센서 4개) → 출력 1개. 숫자 하나만 바뀜. w 4개 + b 1개
opt = torch.optim.SGD(model4.parameters(), lr=0.1)
for epoch in range(500):
    opt.zero_grad(); loss = loss_fn(model4(Ztr), Ytr); loss.backward(); opt.step()   # 네 줄을 세미콜론으로 한 줄에 (같은 코드)

def R2(y, yhat):                                           # 02 의 R² 을 텐서용으로
    return (1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()).item()

with torch.no_grad():
    print("\n[4] 다변수 회귀 — train R²", round(R2(Ytr, model4(Ztr)), 4), "/ test R²", round(R2(Yte, model4(Zte)), 4))
    print("    표준화 가중치:", {k: round(v, 3) for k, v in zip(특징이름, model4.weight.squeeze(0).tolist())})
print("    04 사이킷런: train 0.8155 / test 0.7425, 공기온도 1.986 ← 같은 자리")


# =====================================================================
# 5. 분류 — 03 을 파이토치로: 손실 이름 하나만 바뀐다
# =====================================================================
yc = df["고장여부"].values
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, yc, test_size=0.3, random_state=3, stratify=yc)   # 04 4번와 같은 분할
scaler_c = StandardScaler().fit(Xc_train)
Zc_tr, Zc_te = to_t(scaler_c.transform(Xc_train)), to_t(scaler_c.transform(Xc_test))
Yc_tr = to_t(yc_train).reshape(-1, 1)                     # 0/1 정답도 float32 로! (정수면 손실 함수가 에러)

torch.manual_seed(0)
clf = torch.nn.Linear(4, 1)
loss_c = torch.nn.BCEWithLogitsLoss()                      # ← 회귀와 다른 유일한 줄
opt = torch.optim.SGD(clf.parameters(), lr=0.5)            # 03 과 같은 lr, 같은 걸음 수
for epoch in range(2000):
    opt.zero_grad(); loss = loss_c(clf(Zc_tr), Yc_tr); loss.backward(); opt.step()

with torch.no_grad():
    p = torch.sigmoid(clf(Zc_te)).squeeze(1).numpy()       # 직선값 → sigmoid → 고장 확률. 04 의 predict_proba[:, 1] 을 직접 만든 것
print("\n[5] 분류 — 시험용 고장 확률 상위 5:", np.sort(p)[::-1][:5].round(3))
from sklearn.metrics import recall_score, precision_score      # 채점은 사이킷런 함수를 그대로 빌려 씀
for th in [0.5, 0.2]:
    판정 = (p >= th).astype(int)
    print(f"    임계값 {th}: 재현율 {recall_score(yc_test, 판정, zero_division=0):.2f}  정밀도 {precision_score(yc_test, 판정, zero_division=0):.2f}")


# =====================================================================
# 6. 미니배치 — 데이터가 클 때 조금씩 나눠 걷기 (파이토치를 쓰는 두 번째 이유)
# =====================================================================
from torch.utils.data import TensorDataset, DataLoader

y_m, y_s = Ytr.mean(), Ytr.std()
Ytr_s, Yte_s = (Ytr - y_m) / y_s, (Yte - y_m) / y_s

loader = DataLoader(TensorDataset(Ztr, Ytr_s), batch_size=16, shuffle=True)   # 16대씩, 매 에폭 새로 섞어서
print("\n[6] 미니배치 — 한 에폭에", len(loader), "걸음 (140대 ÷ 16)")

torch.manual_seed(0)
model_mb = torch.nn.Linear(4, 1)
opt = torch.optim.Adam(model_mb.parameters(), lr=0.01)     # Adam: 손잡이마다 보폭을 알아서 조절. 실무에서 가장 흔한 선택. lr 은 0.001~0.01 이 관례
for epoch in range(100):
    for zb, yb in loader:                                  # ← 미니배치 루프. zb = 이번 16대의 센서, yb = 그 정답. 안쪽 네 줄은 그대로
        opt.zero_grad(); loss = loss_fn(model_mb(zb), yb); loss.backward(); opt.step()
with torch.no_grad():
    print("    미니배치 + Adam — test R²:", round(R2(Yte_s, model_mb(Zte)), 4), "(같은 바닥에 도착)")


# =====================================================================
# 7. 다음 과정 예고 — 층을 쌓으면 신경망 (지금은 한 줄만, 코드 없음)
# =====================================================================
# 8. 정리 — 언제 무엇을 쓰나
# =====================================================================
print("""
[8] 손코드 ↔ 사이킷런 ↔ 파이토치
    밟아보기(기울기)     ↔ (내부)               ↔ loss.backward()   (autograd)
    w = w − lr·기울기    ↔ (내부)               ↔ opt.step()        (옵티마이저)
    for epoch 루프       ↔ (내부, fit 한 줄)    ↔ 직접 씀 (zero_grad → loss → backward → step)
    sigmoid + 로그손실   ↔ LogisticRegression  ↔ BCEWithLogitsLoss
    (없음)               ↔ (없음)               ↔ DataLoader 미니배치
    섞고 나누기·표준화   ↔ train_test_split·StandardScaler ↔ (사이킷런 것을 빌려 씀)
    채점                 ↔ metrics             ↔ (사이킷런 것을 빌려 씀)

    선택 기준 한 줄:
      표 데이터, 선형모델/트리, 빨리 결과                → 사이킷런  (fit 한 줄)   ← 이 과정의 데이터는 전부 여기
      아주 큰 데이터, 학습 과정을 직접 조절, (다음 과정) 층 쌓기 → 파이토치  (루프를 내 손에)
      둘 다 심장은 01 의 경사하강. 도구는 원리를 빠르게 할 뿐, 대신하지 않습니다.
""")

# =====================================================================
# 실습 — 몇 걸음이면 충분한가
# =====================================================================
# [문제] 4번에서 500걸음을 걸었습니다. 5걸음, 50걸음, 500걸음을 비교해 보세요.
# 각각 test R2 가 얼마인가요?
#
#   힌트: 매번 torch.manual_seed(0) 으로 시작 위치를 같게 맞춰야 공정한 비교가 됩니다.
