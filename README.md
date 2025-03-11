# Biodegradation Dashboard

This project provides an interactive dashboard for modeling and comparing the biodegradation of plastics under different environmental conditions using multiple microorganisms. It is built with Streamlit and uses various regression and interpolation techniques (including Weibull, Log-Transform Polynomial, PCHIP, and Isotonic Regression) to fit degradation data. The dashboard also computes evaluation metrics (RMSE, R², and MAE) to assess the quality of each model fit.

## Features

- **Data Loading & Cleaning:**  
  Reads and cleans experimental data from a CSV file.  
  Required columns: `plastic_type`, `microorganism`, `degradation_rate`, `days`, `temperature`, and `pH`.

- **Model Fitting:**  
  Uses a fallback chain to automatically select the best fitting model:
  - Direct line (for exactly 2 data points)
  - Weibull model
  - Log-Transform Polynomial model
  - PCHIP interpolation
  - Isotonic regression

- **Optional Log-Fit Curve:**  
  Offers an option to overlay a logarithmic fit of the fallback curve.

- **Evaluation Metrics:**  
  Computes RMSE, R², and MAE for each microorganism’s model to quantify model performance.

- **Multi-Microorganism Comparison:**  
  Compare between 1 to 5 different microorganisms for the same plastic, with separate curves and a final overlay graph.  
  The best-performing microorganism (i.e., the one with the highest predicted degradation at a user-specified day) is highlighted.

- **Sample Data Included:**  
  The project comes with sample data in `experimental_data.csv`. However, for more accurate and meaningful results, it is recommended to use real experimental data.



Please make sure you have Python installed (version 3.7 or later is recommended).

pip install streamlit pandas numpy plotly scipy scikit-learn
