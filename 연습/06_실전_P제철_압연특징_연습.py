import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, recall_score, precision_score

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "수업용_데이터")


r = pd.read_csv(os.path.join(DATA, "T-CR1-SPM01_압연특징.csv"), encoding="utf-8-sig")
print(f"모양 {r.shape}")
print(f"열 {list(r.columns)}")


# =====================================================================
print("1. 열어 보기")

r["시각"] = pd.to_datetime(r["MEAS_DT"])
print(f"기간 {r['시각'].min()} ~ {r['시각'].max()}")

날짜 = r["시각"].dt.normalize().drop_duplicates()
전체일 = pd.date_range(날짜.min(), 날짜.max())
빈날 = 전체일.difference(날짜)
print(f"데이터가 있는 날 {len(날짜)} / 빈 날 {len(빈날)}")

일별 = r.set_index("시각")["VIB-BOT_RMS"].resample("D").mean()
채운것 = 일별.fillna(0)
print(f"빈 날을 0으로 채우면 생기는 일")
print(f"빈 날 빼고 계산한 평균 진동 {일별.mean():.4f}")
print(f"0으로 채우고 계산한 평균 진동 {채운것.std():.4f}")
print(f"{(1 - 채운것.mean() / 일별.mean()) * 100:.0f}% 낮아짐")
print(
    f"빈 날 빼고 계산한 표준편차 {일별.std():.4f} / 0으로 채운 표준편차 {채운것.mean():.4f}"
)

# 기간 2024-01-10 00:00:00 ~ 2024-12-15 02:52:31.448000
# 데이터가 있는 날 314 / 빈 날 27
# 빈 날을 0으로 채우면 생기는 일
# 빈 날 빼고 계산한 평균 진동 0.1080
# 0으로 채우고 계산한 평균 진동 0.0994
# 8% 낮아짐
# 빈 날 빼고 계산한 표준편차 0.0479 / 0으로 채운 표준편차 0.0994

print("\n채널별 RMS 요약")
print(r[["VIB-TOP_RMS", "VIB-BOT_RMS", "CUR-MTR_RMS"]].describe().round(3))

# 채널별 RMS 요약
#        VIB-TOP_RMS  VIB-BOT_RMS  CUR-MTR_RMS
# count      570.000      570.000      570.000
# mean         0.077        0.107      111.292
# std          0.075        0.065       38.382
# min          0.016        0.021        6.325
# 25%          0.048        0.051       82.245
# 50%          0.069        0.090       93.382
# 75%          0.084        0.166      145.054
# max          0.835        0.446      193.420


# =====================================================================
print("\n2. 회귀 문제 정의")

RMS열 = ["VIB-TOP_RMS", "VIB-BOT_RMS", "CUR-MTR_RMS"]
print("RMS끼리 상관계수")
print(f"{r[RMS열].corr().round(2)}")

print("전류 통계끼리 상관(RMS·PTP·STD 가 거의 1 = 중복)")
print(r[[c for c in r.columns if c.startswith("CUR-MTR")]].corr().round(2))

전류열 = ["CUR-MTR_RMS", "CUR-MTR_KUR", "CUR-MTR_CRF"]
X = r[전류열].values
y = r["VIB-BOT_RMS"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

단일 = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr[:, [0]], y_tr)
다변수 = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)

print(f"전류 RMS 하나로 test R2 {단일.score(X_te[:, [0]], y_te):.3f}")
print(f"전류 통계 3개로 test R2 {다변수.score(X_te, y_te):.3f}")
print(f"학습용 R2 {다변수.score(X_tr, y_tr):.3f}")

cv = cross_val_score(make_pipeline(StandardScaler(), LinearRegression()), X, y, cv=5)
print(f"교차검증 5조각 R2 {cv.round(2)} -> 평균 {round(cv.mean(), 3)}")

w = 다변수.named_steps["linearregression"].coef_
print(
    "표준화 가중치 ",
    {c.replace("CUR-MTR_", ""): round(float(v), 3) for c, v in zip(전류열, w)},
)

# 2. 회귀 문제 정의
# RMS끼리 상관계수
#              VIB-TOP_RMS  VIB-BOT_RMS  CUR-MTR_RMS
# VIB-TOP_RMS         1.00         0.59         0.09
# VIB-BOT_RMS         0.59         1.00         0.74
# CUR-MTR_RMS         0.09         0.74         1.00
# 전류 통계끼리 상관(RMS·PTP·STD 가 거의 1 = 중복)
#              CUR-MTR_RMS  CUR-MTR_KUR  CUR-MTR_CRF  CUR-MTR_PTP  CUR-MTR_STD
# CUR-MTR_RMS         1.00        -0.25        -0.09         0.98         1.00
# CUR-MTR_KUR        -0.25         1.00         0.74        -0.15        -0.24
# CUR-MTR_CRF        -0.09         0.74         1.00         0.06        -0.08
# CUR-MTR_PTP         0.98        -0.15         0.06         1.00         0.98
# CUR-MTR_STD         1.00        -0.24        -0.08         0.98         1.00
# 전류 RMS 하나로 test R2 0.638
# 전류 통계 3개로 test R2 0.788
# 학습용 R2 0.626
# 교차검증 5조각 R2 [0.38 0.79 0.52 0.83 0.46] -> 평균 0.595
# 표준화 가중치  {'RMS': 0.05, 'KUR': 0.015, 'CRF': 0.01}


# =====================================================================
print("\n3. 분류 문제 정의")

경계 = r["VIB-TOP_RMS"].quantile(0.90)
r["고진동"] = (r["VIB-TOP_RMS"] > 경계).astype(int)
print(
    f"고진동 경계 {경계:.4f} -> 고진동 {r['고진동'].sum()}건 / {len(r)}건 {r['고진동'].mean():.0%} <- 불균형"
)

입력열 = [c for c in r.columns if c.startswith("VIB-BOT") or c.startswith("CUR-MTR")]
Xc = r[입력열].values
yc = r["고진동"].values
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.3, random_state=0, stratify=yc
)

clf = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
)
clf.fit(Xc_tr, yc_tr)
p = clf.predict_proba(Xc_te)[:, 1]

print(
    classification_report(
        yc_te, clf.predict(Xc_te), target_names=["정상", "고진동"], zero_division=0
    )
)

print("임계값을 바꾸면(재현율과 정밀도를 맞바꿈)")
for th in [0.5, 0.3, 0.7]:
    판정 = (p >= th).astype(int)
    print(
        f"{th}: 재현율 {recall_score(yc_te, 판정):.2f} 정밀도 {precision_score(yc_te, 판정, zero_division=0):.2f}"
    )

w = clf.named_steps["logisticregression"].coef_[0]

상위 = sorted(zip(입력열, w), key=lambda t: -abs(t[1]))[:4]

print(
    f"단서가 된 입력(표준화 가중치 큰 순) {[(c, round(float(v), 2)) for c, v in 상위]}"
)

# 3. 분류 문제 정의
# 고진동 경계 0.0961 -> 고진동 57건 / 570건 10% <- 불균형
#               precision    recall  f1-score   support

#           정상       0.97      0.76      0.85       154
#          고진동       0.26      0.76      0.39        17

#     accuracy                           0.76       171
#    macro avg       0.61      0.76      0.62       171
# weighted avg       0.90      0.76      0.80       171

# 임계값을 바꾸면(재현율과 정밀도를 맞바꿈)
# 0.5: 재현율 0.76 정밀도 0.26
# 0.3: 재현율 0.82 정밀도 0.22
# 0.7: 재현율 0.59 정밀도 0.34
# 단서가 된 입력(표준화 가중치 큰 순) [('VIB-BOT_STD', 1.34), ('VIB-BOT_RMS', 0.91), ('CUR-MTR_KUR', -0.82), ('CUR-MTR_CRF', 0.79)]


# =====================================================================
print("\n4. 분류 문제 정의")

m = pd.read_csv(os.path.join(DATA, "G-02_정비이력.csv"), encoding="utf-8-sig")
m = m[m["EQP_TAG"] == "P1-CR1-SPM01"].copy()
m["감지시각"] = pd.to_datetime(m["DETC_DT"])

고장정비 = m[m["MNT_TYPE"] != "TBM"].sort_values("감지시각")
print(f"이 설비의 정비 {len(m)}건 중 고장성 정비(BM, CBM) {len(고장정비)}건")

감지 = 고장정비["감지시각"].values


def 다음고장까지_일수(t):
    이후 = 감지[감지 >= np.datetime64(t)]
    return (
        (이후[0] - np.datetime64(t)) / np.timedelta64(1, "D") if len(이후) else np.nan
    )


r["다음고장까지"] = r["시각"].apply(다음고장까지_일수)

r["위험"] = (r["다음고장까지"] <= 3).astype(int)

print(f"고장 3일 전 라벨 : 위험 {r['위험'].sum()}건 / {len(r)}건")

print("위험 vs 평소 - 채널별 RMS 평균")
print(r.groupby("위험")[RMS열].mean().round(4))

Xd = r[[c for c in r.columns if c.startswith(("VIB-", "CUR-"))]].values
yd = r["위험"].values
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
    Xd, yd, test_size=0.3, random_state=0, stratify=yd
)

clf2 = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
).fit(Xd_tr, yd_tr)

판정 = clf2.predict(Xd_te)
print(
    f"test 재현율 {recall_score(yd_te, 판정):.2f} / 정밀도 {precision_score(yd_te, 판정, zero_division=0):.2f}"
)
print(f"아무거나 '위험'이라고 찍었을 때의 정밀도 = 위험 비율 {yd_te.mean():.2f}")

# 4. 분류 문제 정의
# 이 설비의 정비 42건 중 고장성 정비(BM, CBM) 22건
# 고장 3일 전 라벨 : 위험 91건 / 570건
# 위험 vs 평소 - 채널별 RMS 평균
#     VIB-TOP_RMS  VIB-BOT_RMS  CUR-MTR_RMS
# 위험
# 0        0.0771       0.1075     111.1530
# 1        0.0792       0.1074     112.0264
# test 재현율 0.59 / 정밀도 0.19
# 아무거나 '위험'이라고 찍었을 때의 정밀도 = 위험 비율 0.16


# print("""
#     정직한 결론: 정밀도 0.19 는 '찍기'(0.16) 수준 → 이 특징들로는 '고장 3일 전'을 못 알아봅니다. 실패가 아니라 발견입니다.
#     (재현율 0.59 가 높아 보이지만, balanced 로 학습해 '위험'을 남발한 결과라 정밀도와 같이 봐야 합니다 — 03 §7.)
#     가능한 이유 (다음 단계의 질문들):
#       · 3.4초짜리 버스트 요약값엔 며칠 단위의 서서히 생기는 변화가 안 담길 수 있다 → 일 단위로 묶어 추세를 보자
#       · '3일 전'이 틀린 가정일 수 있다 → 7일·1일로 바꿔 보자, 고장모드(FAIL_MODE: BRG 베어링 등)별로 나눠 보자
#       · 정비이력의 감지시각이 진동 변화 시점과 다를 수 있다 (사람이 눈치챈 시각이지 이상이 시작된 시각이 아님)
#       · 애초에 이 교육용 재현 데이터에 그런 신호가 심어져 있지 않을 수 있다 (정답지는 배포되지 않음)
#     라벨을 만들었으면 "그 라벨이 데이터와 정말 관계있나"를 먼저 확인하는 것 — 이게 실전에서 제일 중요한 습관.""")


# =====================================================================
# 실습 — 기준을 바꾸면 문제가 달라진다
# =====================================================================
# [문제] 3번에서 '고진동' 을 상위 10% 로 정했습니다. 상위 5% 로 바꾸면 몇 건이 될까요?
#        경계값과 건수를 두 경우 모두 출력해 비교하세요.
#
#   힌트: r["VIB-TOP_RMS"].quantile(0.95) 가 상위 5% 경계입니다.
#         3번에서 고진동을 정할 때 쓴 열이 VIB-TOP_RMS 였으니 같은 열로 비교합니다.
#
# [정답]
#   for q in [0.90, 0.95]:
#       경계2 = r["VIB-TOP_RMS"].quantile(q)
#       y2 = (r["VIB-TOP_RMS"] > 경계2).astype(int)
#       print(f"상위 {round((1 - q) * 100)}% 기준 {경계2:.4f} -> 고진동 {int(y2.sum())}건 / {len(y2)}건")
#       # round 를 쓴 이유: int((1-0.90)*100) 은 9 가 됩니다. 0.1 이 컴퓨터에선
#       #                 정확히 0.1 이 아니라 9.999... 로 계산되기 때문 (00 §2 의 소수 오차)
#
# [나오는 답]
#   상위 10% 기준 0.0961 -> 고진동 57건 / 570건
#   상위 5% 기준 0.1089 -> 고진동 29건 / 570건
#
# [읽는 법]
#   경계값은 0.0961 에서 0.1089 로 아주 조금 올라갔는데, 건수는 57건에서 29건으로 반토막입니다.
#   여기서 06 의 주제가 다시 나옵니다.
#     10% 라고 정했기 때문에 57건이었고, 5% 라고 정했으면 29건입니다.
#     정답이 데이터에 있던 게 아니라 우리가 만든 겁니다.
#     그래서 이런 걸 보고할 때는 반드시 '어떤 기준으로 정했고 왜 그랬는지' 를 같이 씁니다.
#     숫자만 쓰면 읽는 사람이 그게 객관적인 사실인 줄 압니다.
#   스스로 답해 보세요:
#     불균형이 10% 에서 5% 로 심해지면 분류가 더 어려워질까요, 쉬워질까요?
#     -> 더 어려워집니다. 03 에서 배운 대로 드물수록 잡기 힘들어요.

print("\n실습")

for q in [0.90, 0.95]:
    경계2 = r["VIB-TOP_RMS"].quantile(q)
    y2 = (r["VIB-TOP_RMS"] > 경계2).astype(int)
    print(
        f"상위 {round((1 - q) * 100)}% 기준 {경계2:.4f} -> 고진동 {int(y2.sum())}건 / {len(y2)}건"
    )

# 실습
# 상위 10% 기준 0.0961 -> 고진동 57건 / 570건
# 상위 5% 기준 0.1089 -> 고진동 29건 / 570건