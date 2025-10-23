import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 專案標題與介紹 ---
st.title("🍷 葡萄酒品質預測：從特徵選擇到模型評估")
st.markdown("""
本應用程式將展示一個完整的機器學習專案流程，包括資料準備、特徵選擇、模型建立與模型評估。
1.  **資料載入**: 從網路載入紅酒品質資料集。
2.  **缺失值處理**: 檢查並處理資料中的缺失值。
3.  **離群值偵測與處理**: 使用 IQR 方法進行處理。
4.  **特徵標準化**: 使用 Z-score 標準化對特徵進行縮放。
5.  **資料集切分**: 將資料切分為訓練集與測試集。
6.  **特徵選擇**: 使用 SelectKBest 選擇最重要的特徵。
7.  **模型訓練**: 使用多元線性回歸進行訓練。
8.  **模型評估**: 使用 MAE, MSE, RMSE, R² 評估模型表現。
9.  **視覺化分析**: 透過圖表深入分析模型預測結果。
""")

# --- 1. 資料載入 (Data Loading) ---
st.header("1. 資料載入")
# 使用分號作為分隔符讀取 CSV
try:
    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
    st.success("資料載入成功！")
except Exception as e:
    st.error(f"資料載入失敗，請檢查網路連線或 URL。錯誤訊息：{e}")
    st.stop()


# --- 2. 缺失值處理 (Missing Value Handling) ---
st.header("2. 缺失值處理")
if df.isnull().sum().sum() == 0:
    st.info("此資料集非常乾淨，沒有缺失值。")


# --- 3. 離群值偵測與處理 (Outlier Detection and Handling) ---
st.header("3. 離群值偵測與處理")
st.markdown("使用 IQR 方法將離群值調整到邊界值。")

feature_cols = df.columns.drop('quality')
df_outlier_handled = df.copy()

for col in feature_cols:
    Q1 = df_outlier_handled[col].quantile(0.25)
    Q3 = df_outlier_handled[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_outlier_handled[col] = np.clip(df_outlier_handled[col], lower_bound, upper_bound)
st.success("離群值處理完成！")


# --- 4. 特徵標準化 (Feature Standardization) ---
st.header("4. 特徵標準化")
st.markdown("使用 Z-score 標準化，將所有特徵轉換為平均值為 0、標準差為 1 的分佈。")

X = df_outlier_handled.drop('quality', axis=1)
y = df_outlier_handled['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)


# --- 5. 資料集切分 (Train/Test Split) ---
st.header("5. 資料集切分")
st.markdown("將資料集以 80/20 的比例切分為訓練集和測試集。")

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)


# --- 6. 特徵選擇 (Feature Selection) ---
st.header("6. 特徵選擇 (Feature Selection)")
st.markdown("""
使用 `SelectKBest` 搭配 `f_regression` 檢定來評估每個特徵對目標變數 (`quality`) 的重要性。
- **F-score**: 分數越高，代表該特徵與目標變數的線性關係越強。
- **p-value**: p-value < 0.05 通常表示該特徵具有統計顯著性。
""")

# 讓使用者選擇要保留的特徵數量
k = st.slider("請選擇要保留的特徵數量 (k)", min_value=1, max_value=X_train.shape[1], value=6, step=1)

# 執行特徵選擇
selector = SelectKBest(score_func=f_regression, k=k)
selector.fit(X_train, y_train)

# 顯示分數
scores = pd.DataFrame({
    '特徵': X_train.columns,
    'F-score': selector.scores_,
    'p-value': selector.pvalues_
}).sort_values(by='F-score', ascending=False).reset_index(drop=True)

st.write("各特徵的顯著性分析 (F-score & p-value):")
st.dataframe(scores)

# 取得被選中的特徵
selected_mask = selector.get_support()
selected_features = X_train.columns[selected_mask]
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

st.write(f"當 k={k} 時，被選出的重要特徵為：")
st.info(", ".join(selected_features))


# --- 7. 多元線性回歸模型訓練 (Multiple Linear Regression Training) ---
st.header("7. 多元線性回歸模型訓練")
st.markdown("使用**被選出的特徵**來訓練多元線性回歸模型。")

# 建立並訓練模型
model = LinearRegression()
model.fit(X_train_selected, y_train)
st.success("模型訓練完成！")

# 顯示模型係數與截距
st.subheader("模型係數 (Coefficients) 與截距 (Intercept)")
coeffs = pd.DataFrame(
    model.coef_,
    X_train_selected.columns,
    columns=['係數 (Coefficient)']
)
st.dataframe(coeffs)
st.write(f"**模型截距 (Intercept):** `{model.intercept_:.4f}`")


# --- 8. 模型評估 (Model Evaluation) ---
st.header("8. 模型評估 (Model Evaluation)")
st.markdown("使用訓練好的模型對**測試集**進行預測，並計算評估指標。")

# 進行預測
y_pred = model.predict(X_test_selected)

# 計算指標
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 顯示指標
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.metric("R² (R-squared)", f"{r2:.4f}", help="R² 衡量模型對資料變異的解釋程度，越接近 1 越好。R² 為負值表示模型表現比直接預測平均值還差。")
with col2:
    st.metric("MAE (Mean Absolute Error)", f"{mae:.4f}", help="所有預測誤差絕對值的平均，值越小越好。")
with col3:
    st.metric("MSE (Mean Squared Error)", f"{mse:.4f}", help="所有預測誤差平方的平均，對大誤差的懲罰較重。")
with col4:
    st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.4f}", help="MSE 的平方根，與目標變數單位相同，值越小越好。")

# --- 模型表現分析與改進建議 ---
st.subheader("模型表現分析與改進建議")
st.markdown(f"""
目前的線性回歸模型在測試集上的 **R² 約為 {r2:.2f}**。

#### 模型好壞分析：
- R² 分數衡量了模型可以解釋目標變數（酒的品質）變異的百分比。這個數值通常介於 0 和 1 之間。
- 當前的 R² 分數偏低，這意味著目前的特徵和線性模型只能解釋酒品質約 **{r2:.0%}** 的變異性。換句話說，模型的預測能力相當有限，許多影響品質的因素未能被模型捕捉。

#### 可能的改進方向：
1.  **嘗試非線性模型**：
    - 酒的品質與化學成分之間的關係可能不是簡單的線性關係。可以嘗試更複雜的模型，如 **隨機森林 (Random Forest)** 或 **梯度提升機 (Gradient Boosting)**，它們能更好地捕捉非線性特徵。

2.  **進行特徵工程 (Feature Engineering)**：
    - 創造新的、可能更有預測能力的特徵。例如，可以嘗試生成 `total acidity` (fixed acidity + volatile acidity) 或酒精與硫酸鹽的交互作用項 (`alcohol * sulphates`)。

3.  **將問題重新定義為「分類問題」**：
    - 由於 `quality` 是一個 1 到 10 的整數評分，可以將其視為一個分類問題而非迴歸問題。例如，可以將品質分數 7 分以上定義為「優質酒」，4 分以下為「劣質酒」，其餘為「普通酒」，然後訓練一個分類模型來預測酒的等級。
""")

# --- 9. 模型視覺化分析 (Visual Analysis) ---
st.header("9. 模型視覺化分析")

# 建立一個 Figure 和兩個 Axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# --- 圖 1: 預測值 vs. 實際值 ---
sns.scatterplot(x=y_test, y=y_pred, ax=ax1, alpha=0.6, color='royalblue')
# 加上 45 度參考線 (完美預測線)
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
ax1.set_xlabel("實際品質 (Actual Quality)")
ax1.set_ylabel("預測品質 (Predicted Quality)")
ax1.set_title("預測值 vs. 實際值散佈圖")
ax1.grid(True)

# --- 圖 2: 殘差圖 ---
residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals, ax=ax2, alpha=0.6, color='darkorange')
# 加上 0 線
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
# 計算 95% 預測區間線
std_dev = np.std(residuals)
upper_bound = 1.96 * std_dev
lower_bound = -1.96 * std_dev
ax2.axhline(y=upper_bound, color='g', linestyle='--', lw=1.5, label='95% 預測區間')
ax2.axhline(y=lower_bound, color='g', linestyle='--', lw=1.5)
ax2.set_xlabel("預測品質 (Predicted Quality)")
ax2.set_ylabel("殘差 (Residuals = Actual - Predicted)")
ax2.set_title("殘差圖")
ax2.legend()
ax2.grid(True)


# 在 Streamlit 中顯示圖表
st.pyplot(fig)

# --- 圖表說明 ---
st.subheader("圖表解讀")
st.markdown("""
**1. 預測值 vs. 實際值散佈圖 (左圖)**
- **用途**: 檢視模型預測的準確性。
- **解讀**:
    - 藍點代表測試集中每一個樣本的「實際品質」與「預測品質」。
    - 紅色虛線是「完美預測線」(y=x)。如果所有點都落在這條線上，表示模型預測完全準確。
    - 點越靠近紅色虛線，表示預測越準確。從圖中可以看出，點的分佈比較發散，再次印證了 R² 分數不高的結論。

**2. 殘差圖 (右圖)**
- **用途**: 診斷模型的系統性誤差。一個好的迴歸模型，其殘差應該是隨機分佈的。
- **解讀**:
    - 橫軸是模型的「預測值」，縱軸是「殘差」（實際值 - 預測值）。
    - **紅色虛線 (y=0)**: 代表沒有誤差。點在這條線上方表示模型低估了實際值；在下方表示高估了。
    - **理想情況**: 點應該在 y=0 線上下隨機且均勻地分佈，沒有任何明顯的模式（如 U 型、喇叭型）。
    - **綠色虛線 (95% 預測區間)**: 這兩條線框出了約 95% 的殘差所在的範圍。如果大部分點都在這個區間內，表示模型的預測誤差在一個相對穩定的範圍內。落在區間外的點可被視為「離群預測」，值得進一步探討。
""")
