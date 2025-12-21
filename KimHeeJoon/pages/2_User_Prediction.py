# import streamlit as st
# import numpy as np

# st.title("👤 개별 사용자 이탈 예측")

# with st.form("user_input_form"):
#     tenure = st.number_input("가입 기간 (개월)", 0, 120, 12)
#     activity = st.number_input("최근 30일 활동 횟수", 0, 100, 10)
#     payment = st.number_input("최근 결제 금액", 0, 1_000_000, 30000)
#     plan = st.selectbox("요금제", ["Basic", "Standard", "Premium"])

#     submitted = st.form_submit_button("이탈 확률 예측")

# if submitted:
#     churn_proba = np.random.rand()  # 실제 모델 예측으로 교체
#     risk = "High" if churn_proba >= 0.7 else "Medium" if churn_proba >= 0.4 else "Low"

#     st.success(f"이탈 확률: **{churn_proba:.2%}**")
#     st.warning(f"위험 등급: **{risk}**")
