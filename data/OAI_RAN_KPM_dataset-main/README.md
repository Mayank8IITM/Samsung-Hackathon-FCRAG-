# Time series RAN dataset using OAI KPM xApp
This a RAN time series dataset that was captured using a bare metal OAI experimental setup (CN, gNB, FlexRIC and KPM xApp).

![Dataset_logo](https://github.com/user-attachments/assets/6853a3aa-61cf-4d7d-9c20-b171af44af22)

## 📊 Dataset Overview

This dataset contains Radio Access Network (RAN) metrics collected using the KPM xApp in a 5G emulation environment based on OpenAirInterface (OAI). It includes time-series measurements representing network behavior over 5 weeks (~10,080 samples).

## 🚀 Key Features

- **Source**: Bare-metal OAI real-time experimental setup including:
  - OAI 5G Core Network
  - OAI gNB
  - FlexRIC with KPM xApp
  - OAI RFSim and UEs

- **Traffic Pattern**:
  - iperf3 used with configurable/customizable traffic profiles
  - Data captured using the KPM xApp

- **Metrics Collected**:
  - UE identifiers & registration status
  - Latency
  - PRB allocation (DL & UL)
  - PDCP SDU volumes (DL & UL)
  - RLC SDU delay (DL)
  - UE throughput (DL & UL)

## 🗂️ Data Structure

| Column               | Description                       |
|----------------------|------------------------------------|
| `Register`           | Registration index                 |
| `UE.Id`              | User Equipment identifier          |
| `Latency`            | Network latency                    |
| `RRU.PrbTotDl`       | Downlink PRB usage                 |
| `RRU.PrbTotUl`       | Uplink PRB usage                   |
| `DRB.PdcpSduVolumeDL`| PDCP downlink SDU volume           |
| `DRB.PdcpSduVolumeUL`| PDCP uplink SDU volume             |
| `DRB.RlcSduDelayDl`  | RLC downlink SDU delay             |
| `DRB.UEThpDl`        | UE throughput (DL)                 |
| `DRB.UEThpUl`        | UE throughput (UL)                 |

## 🧪 Traffic Pattern Rationale

- **Weekdays (Mon–Sat)**: Daytime peak traffic, reduced at night
- **Sundays**: Constant high traffic to simulate events (e.g., sports venues)
- Customizable via iperf3 profiles

## 🤖 AI Model Development

The dataset was used to train an LSTM-based AI model adapted from cloud CPU forecasting to predict PRB usage patterns:
**Model Arhcitecture**
- **Task**: Forecast PRB usage
- **Model**: LSTM-based time-series predictor
  - Input: 6 time steps (30 min history)
  - Layers: Single LSTM layer (50 units) + dense output layer
  - Training: Adam optimiser, MAE loss, Softmax activation
  - Epochs: 1000
  - Data split: 70% train/validation, 30% testing

## 🔍 Usage Notes

- Periodic weekly patterns observed
- The traffic generation was made in the Uplink direction (for the PRB usage)
- Throughput and PRB usage show strong correlation
- Dataset is ML-ready (cleaned & structured)

## 📈 Applications

- Forecasting RAN resource usage
- Network slicing control
- FPGA-SoC function reconfiguration
- Proactive 5G management use cases

## 🌱 Future Directions

- Add new RAN metrics
- Explore new AI/ML architectures
- Adapt for different scenarios
- Integrate with broader network management systems

## ⚙️ How to Use the Dataset

### 1. Understand Your Data

Use tools like Pandas to load Excel sheets:

```python
import pandas as pd

kpm_df = pd.read_excel('dataset.xlsx', sheet_name='KPM Metrics')
iperf_df = pd.read_excel('dataset.xlsx', sheet_name='iperf3 Traffic')
````

### 2. Preprocess

- **Load the data:** Use a tool like Python’s Pandas to load your Excel workbooks. For example:
```python
import pandas as pd
kpm_df = pd.read_excel('Copy of 20250321_11.17.57_KPM_vs_iper3_1000ms_05_weeks.xlsx', sheet_name='KPM Metrics')
iperf_df = pd.read_excel('Copy of 20250321_11.17.57_KPM_vs_iper3_1000ms_05_weeks.xlsx', sheet_name='iperf3 Traffic')
````
- **Check and clean data:**
  - Clean missing values, ensure correct data types
  - Create time-based features (day, hour, etc.)
  - Generate lag features and rolling statistics
  - Normalize features for ML models

- **Feature Engineering and Transformation**
  - Time series features: If you have timestamps (or can derive them), create additional features such as time of day, day of week, and seasonal markers. This is especially useful if the traffic follows predictable daily or weekly cycles.
  - Lag features: Create features that represent previous measurements (e.g., the value of PRB usage 5, 10, 15 minutes earlier) to help models learn temporal dependencies.
  - Rolling statistics: Compute moving averages, rolling standard deviations, or differences between consecutive measurements to highlight trends and volatility.
  - Scaling: Normalize or standardize your features. For many ML algorithms—as well as deep learning time series models like LSTM—feature scaling can improve convergence and performance.

### 3. Exploratory Data Analysis (EDA)

- **Visualization:** Plot each metric over time. Use line plots to observe trends and seasonality in PRB usage, latency, and throughput.

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(kpm_df['Timestamp'], kpm_df['RRU.PrbTotDl'], label='Downlink PRBs')
plt.plot(kpm_df['Timestamp'], kpm_df['RRU.PrbTotUl'], label='Uplink PRBs')
plt.legend()
plt.xlabel('Time')
plt.ylabel('PRBs')
plt.title('Time Series of PRB Allocation')
plt.show()
```
- **Correlation Analysis:** Construct a correlation matrix (heatmap) to understand how the different metrics interrelate. This may help in selecting which features to combine when building models.
- **Behavior Under Traffic Conditions:** By overlaying iperf3 traffic measurements with RAN metrics (using a shared time axis) you can assess how controlled traffic affects the network’s performance.


### 4. ML Objectives

- **Forecasting**: Predict future PRB usage (or another RAN metric) based on historical data.
  - **Statistical Models:** ARIMA or exponential smoothing for simpler trends.
  - **Deep Learning Models:**
    - **Recurrent Neural Networks (RNNs) / LSTMs:** Given that the dataset was originally used to retrain an LSTM for forecasting, you can use sliding window approaches (e.g. using sequences of 6 samples over 5-minute intervals) to predict the next value or several steps ahead.
    - **Attention-based or Transformer models:** Can also be explored if the relationships across long sequences are particularly complex.

- **Regression/Anomaly Detection**: You might build a regression model (using tree-based models like XGBoost, Random Forest, or even simple linear regression) to predict one target variable (e.g. PRB usage) from other metrics
  - **Anomaly Detection**: Train unsupervised methods (e.g., Isolation Forest, One-Class SVM, or LSTM-based autoencoders) to identify unusual patterns or outliers in network performance.
- **Clustering**: Identify usage patterns or segments within the data (e.g., peak hours vs. off-peak, or normal vs. abnormal operational modes).
  - Use unsupervised techniques like K-means, DBSCAN, or hierarchical clustering.
  - Dimensionality reduction (e.g., PCA or t-SNE) can visualize high-dimensional relationships.

### 5. Building the Machine Learning Pipeline
- **Data Splitting**
  - **Time-based split:** For forecasting, train on the first 70% of the timeline and test on the remaining 30%. This respects the temporal order of data.
- **Model Training**
  - **For LSTM (Forecasting Example):**
    - Data Transformation: Convert your time series into a supervised learning problem by creating input-output pairs using a sliding window.
    - Model Architecture:
      - Input layer: sequences of, for example, 6 time steps (30 minutes) of several features.
      - LSTM layer: 50 units (or tuned based on your experiment).
      - Dense layer: to generate the final forecast.
    - Training: Use the Adam optimizer and loss functions like Mean Absolute Error (MAE).

Example pseudocode snippet:

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Assume X_train and y_train are built from sliding windows
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(y_train.shape[1]))
model.compile(optimizer='adam', loss='mae')
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=1)
```

- **Model evaluation**
  - **Metrics**
    - For forecasting: Use RMSE, MAE, or MAPE.
    - For regression: Use R², MAE, and RMSE.
    - For classification/anomaly detection: Use precision, recall, and F1-score if you label anomalies.
  - **Visualization of predictions**: Plot the predicted versus actual values to inspect the model’s performance.

### 6. Iterative Optimization and Future Directions
- **Hyperparameter Tuning:** Use grid search or randomized search (for example, through scikit-learn or Keras Tuner) to find the best model parameters.
- **Model Interpretability:** Analyze feature importances (for tree-based models) or attention weights (for sequence models) to understand what drives the predictions.
- **Cross-domain Insights:** The dataset’s combination of RAN and iperf3 traffic can help you not only forecast PRB usage but also study how changes in network traffic from iperf3 measurements) affect the RAN performance.
- **Scalability:** As more metrics become available (or if you generate new datasets under different scenarios), consider building pipelines that support multivariate time-series forecasting.

### 7. Recommended Libraries

* `pandas`, `numpy`
* `matplotlib`, `seaborn`
* `scikit-learn`, `tensorflow`/`keras`
* `statsmodels`, `tsfresh`

## 🧩 Challenges Addressed

* Complex OAI deployment managed with careful setup
* Traffic generation scripted via Python
* Centralized metric collection via KPM xApp

## ✅ Conclusion

This dataset provides a realistic representation of RAN behavior under controlled traffic conditions, enabling the development and validation of AI/ML models for 5G network management. The use of open-source components and customizable traffic patterns makes this approach flexible and adaptable to various research and development scenarios in 5G and beyond.
