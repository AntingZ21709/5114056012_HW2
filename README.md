
# Interactive Regression Analysis Tool

This is a web application built with Python and Streamlit that allows users to interactively explore the properties of linear regression models.

You can dynamically adjust parameters like sample size and data noise, switch between different regression models (OLS, Ridge, Lasso), and visualize the impact of these changes in real-time.

## Features

- **Interactive Controls**: Use sliders in the sidebar to adjust the sample size of the generated data and the level of random noise.
- **Multiple Models**: Choose between three different regression models:
    - **Ordinary Least Squares (OLS)**: A standard linear regression model.
    - **Ridge Regression**: A regularized model useful for preventing overfitting, especially with noisy data.
    - **Lasso Regression**: Another regularized model that can also perform feature selection.
- **Regularization Control**: For Ridge and Lasso, you can adjust the regularization strength (alpha) to see how it affects the model's fit.
- **Rich Visualizations**:
    - **Regression Plot**: See the regression line plotted against the raw data points and the original "true" line.
    - **Outlier Highlighting**: Choose to highlight the top 'k' outliers (points with the largest residuals) directly on the plot.
    - **Residuals Plot**: Analyze the distribution of residuals to assess the model's fit.
- **Performance Metrics**: Key metrics like **R-squared (R²)** and **Mean Squared Error (MSE)** are displayed and updated in real-time.
- **Dynamic Commentary**: The application provides a simple, dynamic discussion of the model's performance based on the current parameters.

## Requirements

The application requires the following Python libraries:

- `streamlit`
- `numpy`
- `pandas`
- `plotly`
- `scipy`
- `scikit-learn`

These are all listed in the `requirements.txt` file.

## Setup and Usage

1.  **Install Dependencies**:
    Open your terminal and navigate to the project directory. Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Application**:
    Once the dependencies are installed, run the following command:
    ```bash
    streamlit run app.py
    ```

3.  **View in Browser**:
    Streamlit will automatically open a new tab in your web browser where you can interact with the application.

