import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.title("Linear Regression Simulator")

# Sidebar for user inputs
st.sidebar.header("Parameters")
a_true = st.sidebar.slider("a (slope)", -10.0, 10.0, 2.5, 0.1)
b_true = st.sidebar.slider("b (intercept)", -10.0, 10.0, 1.0, 0.1)
noise = st.sidebar.slider("Noise intensity (σ)", 0.0, 10.0, 2.0, 0.1)
n_samples = st.sidebar.slider("Sample size (n)", 50, 1000, 200, 50)

# Generate data
@st.cache_data
def generate_data(a, b, noise_std, n):
    x = np.random.rand(n) * 10  # x values from 0 to 10
    epsilon = np.random.normal(0, noise_std, n)
    y = a * x + b + epsilon
    return pd.DataFrame({"x": x, "y": y})

df = generate_data(a_true, b_true, noise, n_samples)

# Fit a linear regression model
# Using numpy's polyfit for simplicity
a_fit, b_fit = np.polyfit(df['x'], df['y'], 1)

# Create the charts
# Scatter plot of the data
scatter_plot = alt.Chart(df).mark_circle(size=60).encode(
    x='x',
    y='y',
    tooltip=['x', 'y']
).properties(
    title="Generated Data"
)

# True line
true_line_df = pd.DataFrame({
    'x': [0, 10],
    'y': [b_true, a_true * 10 + b_true]
})
true_line_chart = alt.Chart(true_line_df).mark_line(color='green', strokeDash=[5,5]).encode(
    x='x',
    y='y'
).properties(
    title="True Line vs. Fitted Line"
)

# Fitted line
fitted_line_df = pd.DataFrame({
    'x': [0, 10],
    'y': [b_fit, a_fit * 10 + b_fit]
})
fitted_line_chart = alt.Chart(fitted_line_df).mark_line(color='red').encode(
    x='x',
    y='y'
)

# Combined chart
st.header("Data and Regression Lines")
st.altair_chart(scatter_plot + true_line_chart + fitted_line_chart, use_container_width=True)

st.write(f"True model: y = {a_true:.2f}x + {b_true:.2f}")
st.write(f"Fitted model: y = {a_fit:.2f}x + {b_fit:.2f}")


# Difference between true and fitted line
st.header("Difference between True and Fitted Line")

x_range = np.linspace(0, 10, 100)
y_true = a_true * x_range + b_true
y_fit = a_fit * x_range + b_fit
difference = y_true - y_fit

diff_df = pd.DataFrame({
    'x': x_range,
    'difference': difference
})

diff_chart = alt.Chart(diff_df).mark_line().encode(
    x='x',
    y=alt.Y('difference', title='Difference (True - Fitted)')
).properties(
    title="Difference between True and Fitted Y values"
)

st.altair_chart(diff_chart, use_container_width=True)
