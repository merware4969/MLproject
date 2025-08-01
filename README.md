아래는 GitHub의 `README.md`에 사용할 수 있도록 구성한 **머신러닝 기반 허리 통증 예측 프로젝트 요약**입니다. 프로젝트 목적, 데이터, 모델링, 성과 및 향후 계획을 포함하여 구조화했습니다.

---

# ML기반 허리 통증 예측 모델 개발 및 시각화

[프로젝트 GitHub 링크](https://github.com/merware4969/MLproject)

## GitHub 디렉토리 구조 및 구성 설명

```
MLproject/
├── dataset/
│   ├── df_cleaned.pkl              # 전처리 완료된 데이터셋
│   ├── optuna_study.pkl            # Optuna 튜닝 결과 (study 객체)
├── models/
│   ├── optuna_lgbm_pipeline.joblib # 최종 LGBM 모델 (SMOTE + 튜닝 포함)
│   ├── logreg_pipeline.joblib      # 비교용 로지스틱 회귀 모델
├── streamlit_app.py                # Streamlit 기반 메인 웹 앱 실행 스크립트
├── test.py                         # 메인 웹 앱 실행 전 테스트 스크립트
├── requirements.txt                # 프로젝트 실행에 필요한 패키지 목록
└── README.md                       # 프로젝트 설명 문서 (본 파일)
```

### 구성 요소 설명

* **dataset/**
  전처리된 데이터와 모델 튜닝 결과가 저장된 폴더입니다. 모델 재현 및 시각화에 활용됩니다.

* **models/**
  학습된 모델 파이프라인을 `.joblib` 형식으로 저장한 폴더입니다. Streamlit 앱에서 불러와 예측에 사용됩니다.

* **app.py**
  사용자 입력 기반 통증 예측 및 시각화를 제공하는 Streamlit 웹 애플리케이션의 메인 파일입니다.

* **requirements.txt**
  프로젝트 실행을 위한 Python 라이브러리 의존성이 정리되어 있습니다.

* **README.md**
  프로젝트 개요 및 개발 내역, 활용 가이드 등이 담긴 문서입니다.

---

## 프로젝트 개요

* **목표**: IPUMS 건강조사 데이터를 활용하여 **허리 통증의 발생 여부를 예측**하는 머신러닝 모델 개발 및 웹 시각화 구현
* **배경**:

  * 근무환경 및 생활습관 변화로 인한 **근골격계 질환 증가**
  * 의료자원의 한계 속에서 **고위험군 조기 식별** 및 선별 도구 필요성 증대
  * 추후 어깨·무릎 통증 등 **다른 부위로 확장 가능한 기반 모델** 확보

## 데이터 출처 및 구성

* **데이터**: [IPUMS NHIS 2023](https://healthsurveys.ipums.org/)
* **행/열 수**: 37,214개 샘플 / 14개 변수
* **주요 변수**:

  * **입력 변수**: 연령, 성별, 인종(RACENEW), 직업/산업군(OCCUPN204, INDSRN204), 주당근로시간(HOURSWRK), 건강 인식(HEALTH), 키/몸무게(HEIGHT/WEIGHT), BMI(BMICALC), 통증이력(PAINARMS3M, PAINLEGS3M)
  * **타겟 변수**: 허리 통증 정도(PAINBACK3M, 1~~4등급) → 이진화: 1=무통증(0), 2~~4=통증(1)

## 주요 작업 흐름

### 1. 데이터 전처리

* 결측치 및 이상치 처리 (CDC 기준 BMI 50 초과 제거 등)
* 명목형 변수(인종, 직업군, 산업군 등) **One-Hot Encoding**
* 수치형 변수 **StandardScaler 적용**
* 종속변수 이진화

### 2. 모델링 및 비교

* **Baseline 모델**: Logistic Regression
* **후보 모델**: RandomForest, XGBoost, LightGBM 등
* **최종 모델**: LightGBM + SMOTE + Optuna 하이퍼파라미터 튜닝

  * 성능지표: AUC=0.677, Recall(민감도)=91.5%
  * 높은 민감도 기반 **스크리닝(선별도구)로 적합**

### 3. 클래스 불균형 대응

* 타겟 클래스 불균형(무통증:통증 = 1:2.1)
* **SMOTE 오버샘플링** 적용 후 성능 개선

### 4. 하이퍼파라미터 튜닝

* `OptunaSearchCV(n_trials=50)` 사용
* 최적 파라미터 예시:

  * `subsample`: 0.9675
  * `colsample_bytree`: 0.6557
  * `reg_lambda`: 0.4454
  * `reg_alpha`: 0.0303



### 5. 웹 서비스 구현 (Streamlit)

* **모델 비교 페이지**: ROC 커브 시각화
* **튜닝 시각화**: Optuna 히스토리, 파라미터 중요도, Contour Plot
* **예측 페이지**: 사용자 입력 기반 통증 예측 결과 출력

## 성과 요약

| 항목           | 내용                                     |
| ------------ | -------------------------------------- |
| 최종 모델        | LightGBM (SMOTE + Optuna 튜닝)           |
| AUC          | 0.677                                  |
| Recall (민감도) | 91.5%                                  |
| 활용           | 선별 검사(screener) 용도 적합                  |
| 웹 앱 구현       | Streamlit 기반, 사용자 입력 예측 및 비교 시각화 기능 포함 |

## 향후 과제 및 개선 방향

* **특이도(정밀도) 개선** 필요: 무통증자를 통증으로 과예측하는 경향 있음
* **예측 임계값 조정**, **클래스 불균형 보정**, **의료 피드백 기반 변수 강화** 예정
* 어깨·무릎 등 **다른 근골격계 통증 예측 모델로 확장 적용** 가능

---

## 데이터 출처

본 프로젝트는 [IPUMS NHIS 2023](https://healthsurveys.ipums.org/) 데이터를 기반으로 하였습니다.
해당 데이터는 비상업적 연구 및 교육 목적에 한해 사용 가능하며, 사용자는 반드시 IPUMS의 사용 약관을 준수해야 합니다.
자세한 내용은 [IPUMS 사용 조건](https://ipums.org/license.shtml)을 참고하십시오.

> 출처: IPUMS NHIS, Minnesota Population Center. [https://doi.org/10.18128/D070.V10.0](https://doi.org/10.18128/D070.V10.0)

