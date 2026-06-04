import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. بناء وتجهيز الموديل والبيانات ---
@st.cache_resource
def load_and_train_model():
    # قراءة البيانات
    data = pd.read_csv('league_of_legends_data_large.csv')
    X = data.drop('win', axis=1)
    y = data['win']

    # تقسيم البيانات وعمل scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # تحويل البيانات إلى Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    # تعريف بنية الموديل
    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_dim):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    input_dim = X_train_tensor.shape[1]
    model = LogisticRegressionModel(input_dim)
    
    # محاولة تحميل الموديل لو جاهز، أو تدريبه سريعاً لو مش موجود
    try:
        model.load_state_dict(torch.load('logistic_regression_model.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
        # تدريب سريع لـ 200 Epochs عشان السيرفر ما يهنقش
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), 'logistic_regression_model.pth')
        
    model.eval()
    return model, scaler, X.columns.tolist()

# استدعاء تجهيزات الموديل
try:
    opt_model, scaler, feature_names = load_and_train_model()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

# --- 2. تصميم واجهة المستخدم باستخدام Streamlit ---
st.set_page_config(page_title="LoL Match Predictor", page_icon="🏆", layout="centered")

st.title("🏆 League of Legends Match Outcome Predictor")
st.write("مرحباً بك! أدخل بيانات الجيم الحالية للتوقع بالفوز أو الخسارة:")
st.markdown("---")

if not data_loaded:
    st.error("⚠️ لم نتمكن من تشغيل الموديل. يرجى التأكد من رفع ملف البيانات `league_of_legends_data_large.csv` في المستودع على GitHub.")
else:
    st.subheader("📊 أدخل قيم الميزات (Features) الخاصة بالجيم:")
    
    # إنشاء حقول مدخلات ديناميكية بناءً على الأعمدة اللي في الداتا عندك بالظبط
    user_inputs = []
    
    # تقسيم المدخلات على عمودين بشكل منظم
    col1, col2 = st.columns(2)
    for i, col_name in enumerate(feature_names):
        with col1 if i % 2 == 0 else col2:
            val = st.number_input(f"Enter {col_name}", value=0.0, step=1.0)
            user_inputs.append(val)

    st.markdown("---")

    # --- 3. زر التوقع ---
    if st.button("🔮 Predict Match Outcome", use_container_width=True):
        # تحويل المدخلات وعمل Scaling لها بنفس طريقة التدريب
        input_array = np.array([user_inputs], dtype=np.float32)
        input_scaled = scaler.transform(input_array)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        # حساب التوقع من الموديل
        with torch.no_grad():
            prediction = opt_model(input_tensor)
            probability = prediction.item()
            
        # عرض النتيجة النهائية للمستخدم
        if probability >= 0.5:
            st.success(f"🎉 **الفريق الأزرق (Blue Team) هو الأقرب للفوز!** بنسبة احتمال: {probability*100:.2f}%")
        else:
            st.error(f"💀 **الفريق الأزرق (Blue Team) معرض للـ Loss.** بنسبة احتمال فوز: {probability*100:.2f}%")
