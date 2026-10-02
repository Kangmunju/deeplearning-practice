# 시계열 데이터(Time Series Data)
# 주가, 기온, 매출액 등 시간의 흐름에 따라 순서대로 관측되고 기록된 데이터
# 시간 의존성(데이터 간 순서가 중요하며 과거의 값이 미래의 값에 영향을 미침)
# 자기 상관성(이전 시간의 데이터가 현재 데이터와 높은 연관성을 가짐)
# 주요 구성 요소 : 추세(Trend), 계절성(Seasonality), 주기성(Cycle), 불규칙성(Irregularity)


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

# 데이터의 시작 날짜부터 끝 날짜까지 하루도 빠짐없이 연속된 날짜 배열 생성
전체일 = pd.date_range(날짜.min(), 날짜.max())

# difference() : 주로 집합 자료형에서 차집합을 구할 때 사용하는 메서드나 개념을 뜻함
#                첫번째 집합에는 존재하지만 두 번째 집합에는 없는 요소를 반환(- 연산자)
빈날 = 전체일.difference(날짜)
print(f"데이터가 있는 날 {len(날짜)} / 빈 날 {len(빈날)}")

# set_index() : 데이터프레임의 특정 열을 행 인덱스로 지정
# sample() : 시계열 데이터의 시간 간격(주기)을 바꾸는 함수
일별 = r.set_index("시각")["VIB-BOT_RMS"].resample("D").mean()
채운것 = 일별.fillna(0)
print(f"빈 날을 0으로 채우면 생기는 일")
print(f"빈 날 빼고 계산한 평균 진동 {일별.mean():.4f}")
print(f"0으로 채우고 계산한 평균 진동 {채운것.mean():.4f}")
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

# describe() : 데이터프레임의 열별 기초 통계량을 요약해서 보여주는 메서드
#              숫자형 데이터를 기본으로 요약, NaN 자동 제외
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

# corr() : 데이터프레임의 열간 상관계수 계산(-1 ~ 1 사이 값으로 반환)
print(f"{r[RMS열].corr().round(2)}")

print("전류 통계끼리 상관(RMS·PTP·STD 가 거의 1 = 중복)")
print(r[[c for c in r.columns if c.startswith("CUR-MTR")]].corr().round(2))

전류열 = ["CUR-MTR_RMS", "CUR-MTR_KUR", "CUR-MTR_CRF"]
X = r[전류열].values  # 문제로 줄 정보(전류 특징 3개)
y = r["VIB-BOT_RMS"].values  # 맞혀야 할 정답(아래쪽 진동 VIB-BOT_RMS)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# 전류 특징 하나만 사용
# X[:, [0]] : X의 모든 행, 0번째 열 -> CUR-MTR_RMS만 꺼내게 됨
# 전류 RMS 하나만 가지고 아래쪽 진동을 예측하는 회귀모델
단일 = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr[:, [0]], y_tr)

# 세 특징을 전부 사용
다변수 = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)

# 현재 모델이 LinearRegression이기 때문에 score()를 사용하면 R2(결정계수)가 나옴
# score() : 분류모델인 경우 정확도 반환 / 회귀모델인경우 결정계수 반환
print(f"전류 RMS 하나로 test R2 {단일.score(X_te[:, [0]], y_te):.3f}")
print(f"전류 통계 3개로 test R2 {다변수.score(X_te, y_te):.3f}")
print(f"학습용 R2 {다변수.score(X_tr, y_tr):.3f}")

# 전류 특징 3개로 NIB-BOT-RMS를 예측하는 이 선형회귀 모델이 새로운 데이터에서도 좋은 성능인가?
# 검증에 사용하는 데이터 조각을 바꿔가며 교차해서 사용하는 것(검증 방법이 여러개 X)
# 별도의 평가 기준을 따로 지정하지 않았으므로 넘겨준 모델(회귀)의 기본 score()를 사용
cv = cross_val_score(make_pipeline(StandardScaler(), LinearRegression()), X, y, cv=5)

# 조각별 편차가 큼 -> 모델의 성능이 안정적이지 않다는 신호
print(f"교차검증 5조각 R2 {cv.round(2)} -> 평균 {round(cv.mean(), 3)}")
# print(type(cv))  # <class 'numpy.ndarray'>

# named_steps : 파이프라인 내의 각 step에 이름으로 접근할 수 있게 해주는 딕셔너리 형태의 속성
#               함수가 아닌 속성이므로 named_steps['스텝이름'] 형태로 사용(괄호 X)
# 다변수 파이프라인에서 학습이 끝난 LinearRegression 모델만 꺼냄
# coef_ : 주어진 학습 데이터에 대해 모델이 학습을 통해 손실을 작게 만들도록 결정된 가중치
# 선형회귀가 학습하면서 구한 각 입력변수의 가중치(계수, 가중치 w)를 가져옴
w = 다변수.named_steps["linearregression"].coef_
print(
    "표준화 가중치 ",
    {c.replace("CUR-MTR_", ""): round(float(v), 3) for c, v in zip(전류열, w)},
)

# 세 전류 특징 중 이 회귀모델에서는 CUR-MTR_RMS가 가장 큰 가중치를 받음
# 세 가지 X 중에서는 RMS가 변할 때 모델의 예측값이 가장 크게 변함
# 다만 벌써 RMS가 진동의 원인이라고 단정할 수는 없음

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

# 상위 10% 정도의 높은 진동을 고진동으로 정함
r["고진동"] = (r["VIB-TOP_RMS"] > 경계).astype(int)
print(
    f"고진동 경계 {경계:.4f} -> 고진동 {r['고진동'].sum()}건 / {len(r)}건 {r['고진동'].mean():.0%} <- 불균형"
)

# 정답인 고진동(r["고진동"]) 자체를 VIB-TOP_RMS를 보고 만들었음
# VIB-TOP_RMS를 입력에 넣어버리면 정답을 X에 알려주는 것이 되므로 제외
입력열 = [c for c in r.columns if c.startswith("VIB-BOT") or c.startswith("CUR-MTR")]

Xc = r[입력열].values  # 분류 모델에게 줄 문제(입력값)
yc = r["고진동"].values  # 분류 모델이 맞혀야 할 정답
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.3, random_state=0, stratify=yc
)

clf = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
)
clf.fit(Xc_tr, yc_tr)

# predict_proba() : 분류 모델이 각 클래스에 속할 확률을 0과 1 사이의 값으로 반환
# 모델이 테스트 데이터의 각 샘플에 대해 모든 클래스의 확률을 계산
# 현재 모델은 이진분류이므로 [0일 확률, 1일 확률] 형태의 2차원 배열
# [:, 1] : 1번 클래스(고진동, 양성, True)의 확률만 가져옴
p = clf.predict_proba(Xc_te)[:, 1]

# predict() : 학습이 완료된 머신러닝, 딥러닝 모델에 새로운 데이터를 입력해 결과를 예측하는 메서드
# predict()는 최종 예측 클래스 레이블(1차원 배열)을 반환
# predict_proba()는 각 클래스에 속할 확률(2차원 배열)을 반환
# yc_te(실제 정답)와 clf.predict(Xc_te)(모델이 최종적으로 예측한 0/1)을 넣어서 평가
print(
    classification_report(
        yc_te, clf.predict(Xc_te), target_names=["정상", "고진동"], zero_division=0
    )
)

print("임계값을 바꾸면(재현율과 정밀도를 맞바꿈)")

# 고진동의 정의는 상위 10%
# 모델이 계산하고 저장한 고진동 확률인 p를 보고 몇 %부터 고진동이라고 판정할지를 바꿔가며 확인
for th in [0.5, 0.3, 0.7]:
    판정 = (p >= th).astype(int)

    # recall_score() : 실제 고진동 중에서 모델이 고진동이라고 잡은 비율
    # precision_score() : 모델이 고진동이라고 판정한 것 중에서 실제로 고진동인 비율
    print(
        f"{th}: 재현율 {recall_score(yc_te, 판정):.2f} 정밀도 {precision_score(yc_te, 판정, zero_division=0):.2f}"
    )

# 학습이 완료된 LogisticRegression 모델의 각 입력 특징의 가중치(계수)를 가져옴
# 로지스틱회귀의 coef_가 지금 같은 이진 분류에서는 2차원 형태로 들어옴
# [0]으로 안쪽의 첫번째 줄만 꺼내옴 -> 1차원 형태로 반환 (5,)의 형태
w = clf.named_steps["logisticregression"].coef_[0]

# zip(입력열, w)의 데이터 각각의 한 쌍이 t에 들어감
# t[0]에는 입력열, t[1]에는 w
# lambda t: -abs(t[1])은 def 정렬기준(t): return -abs(t[1])과 동일
# 가중치가 큰 값을 찾고 있으므로 abs()로 부호는 두고 크기만 봄
# sorted()는 기본적으로 작은 값에서 큰 값 순서로 정렬하므로 -를 붙여 반대를 구함(큰 값부터 앞으로)
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

# m에 여러 설비의 정비이력이 있으므로 지금 우리가 보고 있는 설비의 정비이력만 골라낼 것
# 안쪽 조건에 의해 True인 행만 m에서 가져옴(P1-CR1-SPM01 설비의 행만 남음)
# 골라낸 결과를 복사해 별도의 데이터 프레임으로 만들고 최종적으로 m에 저장
m = m[m["EQP_TAG"] == "P1-CR1-SPM01"].copy()

# m에서 DETC_DT열을 가져와서 날짜, 시간 자료형으로 변환
# m에 '감지시각'이라는 새로운 열을 만들어서 저장
m["감지시각"] = pd.to_datetime(m["DETC_DT"])

# m에서 TBM이 아닌 정비를 골라냄
# 감지시각 순으로 정렬해 고장정비에 저장
고장정비 = m[m["MNT_TYPE"] != "TBM"].sort_values("감지시각")
print(f"이 설비의 정비 {len(m)}건 중 고장성 정비(BM, CBM) {len(고장정비)}건")

# 고장정비 데이터에서 감지시각만 뽑아 NumPy 배열 감지에 저장
# TBM을 제외한 고장정비의 감지시각들만 모여 있게 됨
감지 = 고장정비["감지시각"].values


def 다음고장까지_일수(t):
    # 감지에 있는 날짜들을 현재 센서 측정시각 t와 하나씩 비교
    # 현재 센서 측정시각 t 이후에 발생한 고장정비 시각들만 골라 이후에 저장
    이후 = 감지[감지 >= np.datetime64(t)]

    # 현재 측정시각 t에서 다음 고장정비까지 남은 일수 반환
    return (
        (이후[0] - np.datetime64(t)) / np.timedelta64(1, "D") if len(이후) else np.nan
    )


r["다음고장까지"] = r["시각"].apply(다음고장까지_일수)

r["위험"] = (r["다음고장까지"] <= 3).astype(int)

print(f"고장 3일 전 라벨 : 위험 {r['위험'].sum()}건 / {len(r)}건")

print("위험 vs 평소 - 채널별 RMS 평균")
print(r.groupby("위험")[RMS열].mean().round(4))

Xd = r[[c for c in r.columns if c.startswith(("VIB-", "CUR-"))]].values

# 위험열을 NumPy 배열로 꺼내서 분류 모델의 정답으로 사용
yd = r["위험"].values

Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
    Xd, yd, test_size=0.3, random_state=0, stratify=yd
)

clf2 = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
).fit(Xd_tr, yd_tr)

# 학습시킨 clf2에게 테스트용 센서 데이터 Xd_te를 주고 0/1로 최종 판정
# 0 = 위험하지 않다고 예측
# 1 = 위험하다고 예측
판정 = clf2.predict(Xd_te)

# recall_score() : 실제로 위험한 것 중 모델이 위험이라고 잡은 비율
# precision_score() : 모델이 위험이라고 판정한 것 중 실제로 위험한 비율
print(
    f"test 재현율 {recall_score(yd_te, 판정):.2f} / 정밀도 {precision_score(yd_te, 판정, zero_division=0):.2f}"
)

# 정답을 0과 1로 만들어놨으므로 평균만 내도 1의 비율을 구할 수 있음
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

print("\n 실습")

for th2 in [0.90, 0.95]:
    경계2 = r["VIB-TOP_RMS"].quantile(th2)
    r2 = (r["VIB-TOP_RMS"] > 경계2).astype(int)
    print(f"상위 {(1 - th2):.0%} 기준 {경계2:.4f} -> 고진동 {r2.sum()}건 / {len(r2)}건")
