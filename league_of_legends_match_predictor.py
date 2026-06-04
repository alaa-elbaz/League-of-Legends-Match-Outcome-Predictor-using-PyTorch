import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# --- 1. بناء وتجهيز الموديل والبيانات ---
@st.cache_resource
def load_and_train_model():
    data = pd.read_csv('league_of_legends_data_large.csv')
    X = data.drop('win', axis=1)
    y = data['win']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_dim):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    input_dim = X_train_tensor.shape[1]
    model = LogisticRegressionModel(input_dim)
    
    try:
        model.load_state_dict(torch.load('logistic_regression_model.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), 'logistic_regression_model.pth')
        
    model.eval()
    return model, scaler, X.columns.tolist(), X_test_tensor, y_test_tensor

try:
    opt_model, scaler, feature_names, X_test_tensor, y_test_tensor = load_and_train_model()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

# --- 2. تصميم واجهة المستخدم ---
st.set_page_config(page_title="LoL Match Predictor", page_icon="🏆", layout="centered")

st.title("🏆 League of Legends Match Outcome Predictor")
st.markdown("---")

if not data_loaded:
    st.error("⚠️ لم نتمكن من تشغيل الموديل. يرجى التأكد من رفع ملف البيانات `league_of_legends_data_large.csv` في المستودع.")
else:
    # تقسيم الصفحة لتبويبين (Tab للتوقع و Tab للرسومات والتحليلات)
    tab1, tab2 = st.tabs(["🔮 Match Prediction", "📊 Model Analytics & Charts"])
    
    with tab1:
        st.subheader("📊 أدخل قيم الميزات الحالية للجيم:")
        user_inputs = []
        col1, col2 = st.columns(2)
        for i, col_name in enumerate(feature_names):
            with col1 if i % 2 == 0 else col2:
                val = st.number_input(f"Enter {col_name}", value=0.0, step=1.0, key=f"input_{col_name}")
                user_inputs.append(val)

        st.markdown("---")

        if st.button("🔮 Predict Match Outcome", use_container_width=True):
            input_array = np.array([user_inputs], dtype=np.float32)
            input_scaled = scaler.transform(input_array)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
            
            with torch.no_grad():
                prediction = opt_model(input_tensor)
                probability = prediction.item()
                
            if probability >= 0.5:
                st.success(f"🎉 **الفريق الأزرق (Blue Team) هو الأقرب للفوز!** بنسبة احتمال: {probability*100:.2f}%")
            else:
                st.error(f"💀 **الفريق الأزرق (Blue Team) معرض للـ Loss.** بنسبة احتمال فوز: {probability*100:.2f}%")

    with tab2:
        st.subheader("📈 التحليلات والرسومات البيانية للموديل")
        st.write("هنا مصفوفة الارتباك وأهمية المتغيرات التي كانت مخفية في الكود:")
        
        # 1. رسم الـ Confusion Matrix
        st.write("#### 🔹 Confusion Matrix")
        with torch.no_grad():
            test_outputs = opt_model(X_test_tensor)
            test_preds = (test_outputs >= 0.5).float()
        
        cm = confusion_matrix(y_test_tensor.numpy(), test_preds.numpy())
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['Predicted Loss', 'Predicted Win'],
                    yticklabels=['Actual Loss', 'Actual Win'])
        ax1.set_title('Confusion Matrix')
        st.pyplot(fig1)
        
        st.markdown("---")
        
        # 2. رسم أهمية المتغيرات (Feature Importance)
        st.write("#### 🔹 Feature Importance (Weights)")
        weights = opt_model.linear.weight.data.numpy().flatten()
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance (Weight)': weights
        }).sort_values(by='Importance (Weight)', ascending=True)
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = ['red' if w < 0 else 'green' for w in feature_importance_df['Importance (Weight)']]
        ax2.barh(feature_importance_df['Feature'], feature_importance_df['Importance (Weight)'], color=colors)
        ax2.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax2.set_title('Feature Importance from Model')
        plt.tight_layout()
        st.pyplot(fig2)
