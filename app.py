import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Set a fixed random seed for reproducibility
np.random.seed(42)

# --- App Title and Description ---
st.set_page_config(layout="wide")
st.title('Interactive Regression Analysis')
st.write('Explore different regression models and see how they are affected by sample size, noise, and regularization.')

# --- Sidebar Controls ---
st.sidebar.header('Simulation Controls')
sample_size = st.sidebar.slider('Sample Size', min_value=10, max_value=1000, value=100, step=10)
noise = st.sidebar.slider('Noise Level', min_value=0.0, max_value=20.0, value=2.0, step=0.5)

st.sidebar.header('Model Selection')
model_type = st.sidebar.selectbox(
    'Choose Regression Model',
    ('Ordinary Least Squares', 'Ridge', 'Lasso')
)

alpha = 1.0
if model_type in ['Ridge', 'Lasso']:
    alpha = st.sidebar.slider(
        'Regularization Strength (alpha)',
        min_value=0.01, max_value=100.0, value=1.0, step=0.1, format="%.2f"
    )

# --- Data Generation ---
X = np.linspace(-5, 5, sample_size)
original_y = 2 * X + 1
epsilon = np.random.normal(0, noise, size=sample_size)
y_noisy = original_y + epsilon
X_reshaped = X.reshape(-1, 1)

# --- Model Training ---
if model_type == 'Ordinary Least Squares':
    model = LinearRegression()
elif model_type == 'Ridge':
    model = Ridge(alpha=alpha)
else: # Lasso
    model = Lasso(alpha=alpha)

model.fit(X_reshaped, y_noisy)
y_pred = model.predict(X_reshaped)

# --- DataFrame and Metrics ---
df = pd.DataFrame({
    'X': X,
    'Y_Original': original_y,
    'Y_Noisy': y_noisy,
    'Y_Predicted': y_pred,
})
df['Residuals'] = df['Y_Noisy'] - df['Y_Predicted']
df['Abs_Residuals'] = abs(df['Residuals'])

mse = mean_squared_error(df['Y_Noisy'], df['Y_Predicted'])
r2 = r2_score(df['Y_Noisy'], df['Y_Predicted'])

# --- Main Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"{model_type} Regression Analysis")
    # --- Outlier Controls ---
    st.sidebar.header("Outlier Analysis")
    show_outliers = st.sidebar.checkbox("Highlight Top-k Outliers")
    k_outliers = 0
    if show_outliers:
        k_outliers = st.sidebar.slider('Number of outliers (k)', min_value=1, max_value=20, value=5, step=1)
    
    df_sorted = df.sort_values(by='Abs_Residuals', ascending=False)
    outliers = df_sorted.head(k_outliers)

    # --- Main Regression Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['X'], y=df['Y_Noisy'], mode='markers', name='Data Points', marker=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=df['X'], y=df['Y_Predicted'], mode='lines', name=f'{model_type} Line', line=dict(color='#ff7f0e', width=3)))
    fig.add_trace(go.Scatter(x=df['X'], y=df['Y_Original'], mode='lines', name='Original (True) Line', line=dict(color='#2ca02c', dash='dash')))
    
    if show_outliers and k_outliers > 0:
        fig.add_trace(go.Scatter(
            x=outliers['X'], y=outliers['Y_Noisy'], mode='markers',
            name='Top-k Outliers',
            marker=dict(color='#d62728', size=12, symbol='x')
        ))

    fig.update_layout(title='Regression Model Fit', xaxis_title='X', yaxis_title='Y', legend_title='Legend')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("Model Performance")
    st.metric(label="R-squared (R²)", value=f"{r2:.3f}")
    st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.3f}")
    
    st.subheader("Performance Discussion")
    r2_desc = "excellent" if r2 > 0.9 else "good" if r2 > 0.7 else "moderate" if r2 > 0.4 else "poor"
    st.markdown(f"- The **R² value of {r2:.3f}** indicates a **{r2_desc}** fit. This means the model explains approximately **{r2:.1%}** of the variance in the data.")
    st.markdown(f"- The **MSE of {mse:.3f}** measures the average squared difference between the observed and predicted values. Lower is better.")
    if noise > 5 and sample_size < 100:
        st.markdown("- **High noise** and a **small sample size** are challenging this model, likely reducing its R² and increasing its MSE.")
    if model_type == 'Ridge' and alpha > 10 and noise > 5:
        st.markdown(f"- With high noise, the **Ridge model (alpha={alpha})** is pulling the regression line away from potential outliers, creating a more generalized (but potentially less accurate on this sample) fit.")
    if model_type == 'Lasso' and alpha > 1 and noise > 5:
        st.markdown(f"- The **Lasso model (alpha={alpha})** is also using regularization. At high alpha values, it can even reduce some coefficients to zero, simplifying the model.")

# --- Detailed Analysis Tabs ---
st.header("Detailed Analysis")
tab1, tab2, tab3 = st.tabs(["Residuals Plot", "Data & Residuals Table", "Top-k Outliers Table"])

with tab1:
    st.subheader("Residuals vs. X")
    st.write("In a well-fitting model, residuals should be randomly scattered around the red dashed line (y=0) with no obvious pattern.")
    resid_fig = go.Figure()
    resid_fig.add_trace(go.Scatter(x=df['X'], y=df['Residuals'], mode='markers', name='Residuals'))
    resid_fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(resid_fig, use_container_width=True)

with tab2:
    st.subheader("Data and Calculated Residuals")
    st.dataframe(df[[ 'X', 'Y_Noisy', 'Y_Predicted', 'Residuals']].head(20))

with tab3:
    st.subheader(f"Top {k_outliers} Outliers")
    if show_outliers and k_outliers > 0:
        st.dataframe(outliers)
    else:
        st.info("Enable and select k in the sidebar under 'Outlier Analysis' to see results here.")