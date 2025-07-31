import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# ------------------------------
# 파일 경로
# ------------------------------
DATA_PATH = 'dataset/df_cleaned.pkl'
LGBM_PATH = 'models/optuna_lgbm_pipeline.joblib'
LOGREG_PATH = 'models/logreg_pipeline.joblib'

# ------------------------------
# 데이터 및 모델 로딩
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_pickle(DATA_PATH)
    df['PAINBACK3M'] = df['PAINBACK3M'].apply(lambda x: 0 if x == 1 else 1)
    return df

@st.cache_resource
def load_models():
    model_lgbm = joblib.load(LGBM_PATH)
    model_logreg = joblib.load(LOGREG_PATH)
    return model_lgbm, model_logreg

df = load_data()
model_lgbm, model_logreg = load_models()

# race_group 제거된 변수 목록
numerical_cols = ['HOURSWRK', 'HEIGHT', 'WEIGHT', 'BMICALC']
categorical_cols = ['Occ_Group', 'Ind_Group', 'PAINARMS3M', 'PAINLEGS3M', 'HEALTH']
X = df[numerical_cols + categorical_cols + ['race_group']]
y = df['PAINBACK3M']

# ------------------------------
# 사이드바 메뉴
# ------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="💡 기능 선택",
        options=["ROC 커브 비교", "통증 예측 시뮬레이션"],
        icons=["bar-chart", "activity"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "1rem"},
            "icon": {"color": "orange", "font-size": "20px"},
            "nav-link-selected": {"background-color": "#f9a825"},
        }
    )

# ------------------------------
# 1. ROC 커브 비교 시각화
# ------------------------------
if selected == 'ROC 커브 비교':
    st.title("📈 모델 성능 비교 (ROC Curve)")

    y_proba_lgbm = model_lgbm.predict_proba(X)[:, 1]
    y_proba_logreg = model_logreg.predict_proba(X)[:, 1]

    fpr_lgbm, tpr_lgbm, _ = roc_curve(y, y_proba_lgbm)
    fpr_logreg, tpr_logreg, _ = roc_curve(y, y_proba_logreg)

    auc_lgbm = auc(fpr_lgbm, tpr_lgbm)
    auc_logreg = auc(fpr_logreg, tpr_logreg)

    fig, ax = plt.subplots()
    ax.plot(fpr_lgbm, tpr_lgbm, label=f'LGBM (AUC = {auc_lgbm:.2f})', linewidth=2)
    ax.plot(fpr_logreg, tpr_logreg, label=f'LogReg (AUC = {auc_logreg:.2f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc='lower right')
    st.pyplot(fig)

# ------------------------------
# 2. 사용자 입력 → 통증 예측
# ------------------------------
else:
    st.title("🩺 통증 예측 시뮬레이션 (LGBM 모델)")

    # 범주형 변수 옵션
    occ_options = sorted(df['Occ_Group'].dropna().unique())
    ind_options = sorted(df['Ind_Group'].dropna().unique())
    arms_options = sorted(df['PAINARMS3M'].dropna().unique())
    legs_options = sorted(df['PAINLEGS3M'].dropna().unique())
    health_options = sorted(df['HEALTH'].dropna().unique())

    # 직업군 산업군 설명
    occ_ind_labels = {
        "Active": "활동적인 직업 및 산업군",
        "Mixed": "혼합형 직업 및 산업군(교사, 요리사, 간호사)",
        "Sedentary": "앉아서 하는 직업 및 산업군"
    }
    
    occ_display = [occ_ind_labels[val] for val in occ_options if val in occ_ind_labels]
    ind_display = [occ_ind_labels[val] for val in ind_options if val in occ_ind_labels]

    # 통증 레벨 설명
    pain_level_labels = {
        1: "1 - 전혀 없음",
        2: "2 - 약간 있음",
        3: "3 - 많이 있음",
        4: "4 - 중간 정도"
    }

    arms_display = [pain_level_labels[val] for val in arms_options if val in pain_level_labels]
    legs_display = [pain_level_labels[val] for val in legs_options if val in pain_level_labels]

    with st.form("user_input_form"):
        st.subheader("사용자 입력")

        col1, col2 = st.columns(2)
        with col1:
            hourswrk = st.slider("주당 근로시간", 0, 90, 40)
            height = st.slider("신장(cm)", 140, 200, 170)
            weight = st.slider("체중(kg)", 40, 120, 65)
        with col2:
            occ_label = st.selectbox("직업 그룹", occ_display)
            ind_label = st.selectbox("산업 그룹", ind_display)
            pain_arms_label = st.selectbox("최근 팔 통증 정도(3개월)", arms_display)
            pain_legs_label = st.selectbox("최근 다리 통증 정도(3개월)", legs_display)
            health = st.selectbox("자가 건강 인식 정도(1=최고, 5=최악)", health_options)

        # BMI 자동 계산
        bmi = round(weight / ((height / 100) ** 2), 2)
        st.markdown(f"**🧮 자동 계산된 BMI: `{bmi}`**")

        submitted = st.form_submit_button("예측하기")

    # 역매핑
    pain_arms = [k for k, v in pain_level_labels.items() if v == pain_arms_label][0]
    pain_legs = [k for k, v in pain_level_labels.items() if v == pain_legs_label][0]
    
    occ = [k for k, v in occ_ind_labels.items() if v == occ_label][0]
    ind = [k for k, v in occ_ind_labels.items() if v == ind_label][0]

    if submitted:
        input_data = pd.DataFrame([{
            'HOURSWRK': hourswrk,
            'HEIGHT': height,
            'WEIGHT': weight,
            'BMICALC': bmi,
            'Occ_Group': occ,
            'Ind_Group': ind,
            'PAINARMS3M': pain_arms,
            'PAINLEGS3M': pain_legs,
            'HEALTH': health
        }])

        pred = model_lgbm.predict(input_data)[0]
        prob = model_lgbm.predict_proba(input_data)[0][1]

        st.success(f"✅ 예측 결과: {'통증 있음 (1)' if pred == 1 else '무통증 (0)'}")
        st.info(f"📊 예측 확률 (통증일 확률): {prob:.2%}")
