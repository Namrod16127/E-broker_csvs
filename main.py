
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

# A variable for predicting 1 day out into the future
forecast_out = 1
st.title("E-broker")
# Choosing the dataset
dataset_name = st.sidebar.selectbox("Select Dataset",
("BATU", "BOBU", "CENT", "CQCIL", "DFCU", "EABL", "EBL",
 "JHL", "KA", "KCB", "NIC", "NMG", "NVL", "SBU", "UCHM", "UCL", "UMEM"))


def get_dataset(dataset_name):
    if dataset_name == "BATU":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20BATU%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)
    elif dataset_name == "BOBU":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20BOBU%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    elif dataset_name == "CENT":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20CENT%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    elif dataset_name == "CQCIL":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20CQCIL%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    elif dataset_name == "DFCU":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20DFCU%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    elif dataset_name == "EABL":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20EABL%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    elif dataset_name == "EBL":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20EBL%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    elif dataset_name == "JHL":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20JHL%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    elif dataset_name == "KA":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20KA%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    elif dataset_name == "KCB":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20KCB%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    elif dataset_name == "NIC":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20NIC%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    elif dataset_name == "NMG":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20NMG%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    elif dataset_name == "NVL":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20NVL%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    elif dataset_name == "SBU":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20SBU%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    elif dataset_name == "UCHM":
        url = requests.get('https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20UCHM%20data.csv')
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    elif dataset_name == "UCL":
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20UCL%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    else:
        url = 'https://raw.githubusercontent.com/Namrod16127/E-broker_csvs/main/USE%20UMEM%20data.csv'
        data = pd.read_csv(url, index_col='Date', parse_dates=True)

    # Selecting Last Traded Price (UGX) column
    data = data[['Last Traded Price (UGX)']]
    df = data
    # Creating the column predictions
    # Create another column (target the independent variable) shifted 1 units up
    df['Prediction'] = df[['Last Traded Price (UGX)']].shift(-forecast_out)
    # Create the independent data set (X)
    # Convert the dataframe to a numpy array
    x = np.array(df.drop(['Prediction'], 1))
    # Remove the last row
    X = x[:-forecast_out]
    # Create the dependent data set (y)
    # Convert the dataframe to a numpy array (All of the values including NaN's)
    y = np.array(df['Prediction'])
    # Get all of the y values except the last row
    y = y[:-forecast_out]
    return X, y


X, y = get_dataset(dataset_name)
# Split the data into 80% training 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Create and train the Support Vector Machine (SVR)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)
# Testing model:  Score returns the co-efficient of determination R^2 of the prediction
# The best possible score is 1
# May exceed 1 or go below 0. If so refresh
svm_confidence = svr_rbf.score(x_test, y_test)

x_forecast = X[-forecast_out:]
svm_prediction = svr_rbf.predict(x_forecast)
st.write(f"Prediction for next Closing price(SVM) = {svm_prediction}")
st.write(f"Accuracy(SVM) = {svm_confidence}")

# Create and train Linear Regression Model
lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)
# Testing model:  Score returns the co-efficient of determination R^2 of the prediction
# The best possible score is 1
lr_confidence = lr.score(x_test, y_test)

# Show the predictions for the Linear Regression Model for the next 'n' days
lr_prediction = lr.predict(x_forecast)
st.write(f"Prediction for next Closing price(Linear regression) = {lr_prediction}")
st.write(f"Accuracy(Linear regression) = {lr_confidence}")

