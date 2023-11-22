'''
Goal of LSTM microservice:
1. LSTM microservice will accept the GitHub data from Flask microservice and will forecast the data for next 1 year based on past 30 days
2. It will also plot three different graph (i.e.  "Model Loss", "LSTM Generated Data", "All Issues Data") using matplot lib 
3. This graph will be stored as image in Google Cloud Storage.
4. The image URL are then returned back to Flask microservice.
'''
# Import all the required packages
from flask import Flask, jsonify, request, make_response
import os
from dateutil import *
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from flask_cors import CORS

# Tensorflow (Keras & LSTM) related packages
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Facebook Prophet packages
from werkzeug.http import is_resource_modified
import json
import dateutil.relativedelta
from dateutil import *
from datetime import date
import requests
import matplotlib.pyplot as plt
from prophet import Prophet 

# Stats Model Packages
import statsmodels
import statsmodels.api as sm

# Import required storage package from Google Cloud Storage
from google.cloud import storage

# Initilize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)
# Initlize Google cloud storage client
client = storage.Client()

# Add response headers to accept all types of  requests

def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

#  Modify response headers when returning to the origin

def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

'''
API route path is  "/api/forecast"
This API will accept only POST request
'''

@app.route('/api/forecast', methods=['POST'])
def forecast():
    body = request.get_json()
    issues = body["issues"]
    type = body["type"]
    repo_name = body["repo"]
    data_frame = pd.DataFrame(issues)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']

    df['ds'] = df['ds'].astype('datetime64[ns]')
    array = df.to_numpy()
    x = np.array([time.mktime(i[0].timetuple()) for i in array])
    y = np.array([i[1] for i in array])

    lzip = lambda *x: list(zip(*x))

    days = df.groupby('ds')['ds'].value_counts()
    Y = df['y'].values
    X = lzip(*days.index.values)[0]
    firstDay = min(X)

    '''
    To achieve data consistancy with both actual data and predicted values, 
    add zeros to dates that do not have orders
    [firstDay + timedelta(days=day) for day in range((max(X) - firstDay).days + 1)]
    '''
    Ys = [0, ]*((max(X) - firstDay).days + 1)
    days = pd.Series([firstDay + timedelta(days=i)
                      for i in range(len(Ys))])
    for x, y in zip(X, Y):
        Ys[(x - firstDay).days] = y

    # Modify the data that is suitable for LSTM
    Ys = np.array(Ys)
    Ys = Ys.astype('float32')
    Ys = np.reshape(Ys, (-1, 1))
    # Apply min max scaler to transform the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    Ys = scaler.fit_transform(Ys)
    # Divide training - test data with 80-20 split
    train_size = int(len(Ys) * 0.80)
    test_size = len(Ys) - train_size
    train, test = Ys[0:train_size, :], Ys[train_size:len(Ys), :]
    print('train size:', len(train), ", test size:", len(test))

    # Create the training and test dataset
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    '''
    Look back decides how many days of data the model looks at for prediction
    Here LSTM looks at approximately one month data
    '''
    look_back = 30
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Verifying the shapes
    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

    # Model to forecast
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model with training data and set appropriate hyper parameters
    history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

    '''
    Creating image URL
    BASE_IMAGE_PATH refers to Google Cloud Storage Bucket URL.Add your Base Image Path in line 145
    if you want to run the application local
    LOCAL_IMAGE_PATH refers local directory where the figures generated by matplotlib are stored
    These locally stored images will then be uploaded to Google Cloud Storage
    '''
    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-microservice11-bucket/')
    # DO NOT DELETE "static/images" FOLDER as it is used to store figures/images generated by matplotlib
    LOCAL_IMAGE_PATH = "static/images/"

    # Creating the image path for model loss, LSTM generated image and all issues data image
    MODEL_LOSS_IMAGE_NAME = "model_loss_" + type +"_"+ repo_name + ".png"
    MODEL_LOSS_URL = BASE_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME

    LSTM_GENERATED_IMAGE_NAME = "lstm_generated_data_" + type +"_" + repo_name + ".png"
    LSTM_GENERATED_URL = BASE_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME

    ALL_ISSUES_DATA_IMAGE_NAME = "all_issues_data_" + type + "_"+ repo_name + ".png"
    ALL_ISSUES_DATA_URL = BASE_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME

    # Add your unique Bucket Name if you want to run it local
    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'Your_BUCKET_NAME')

    # Model summary()

    # Plot the model loss image
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss For ' + type)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)

    # Predict issues for test data
    y_pred = model.predict(X_test)

    # Plot the LSTM Generated image
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(np.arange(0, len(Y_train)), Y_train, 'g', label="history")
    axs.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)),
             Y_test, marker='.', label="true")
    axs.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)),
             y_pred, 'r', label="prediction")
    axs.legend()
    axs.set_title('LSTM Generated Data For ' + type)
    axs.set_xlabel('Time Steps')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME)

    # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('All Issues Data')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME)

    # Uploads an images into the google cloud storage bucket
    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(MODEL_LOSS_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)
    new_blob = bucket.blob(ALL_ISSUES_DATA_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME)
    new_blob = bucket.blob(LSTM_GENERATED_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME)

    # Construct the response
    json_response = {
        "model_loss_image_url": MODEL_LOSS_URL,
        "lstm_generated_image_url": LSTM_GENERATED_URL,
        "all_issues_data_image": ALL_ISSUES_DATA_URL
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)


    @app.route('/api/statmisc', methods=['POST'])
def statistical_analysis():
    """
    Handles POST request to perform statistical analysis on GitHub issues data.
    Generates observation and forecast plots for the provided data.
    """

    # Extract data from the request body
    request_data = request.get_json()
    analysis_type = request_data["type"]
    repository_name = request_data["repo"]
    issues_data = request_data["issues"]

    # Set up environment variables for image paths
    base_image_path = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path')
    bucket_name = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')
    local_image_path = "static/images/"

    # Generate filenames for observation and forecast images
    observation_image_name = f"stats_observation_{analysis_type}_{repository_name}.png"
    forecast_image_name = f"stats_forecast_{analysis_type}_{repository_name}.png"

    # Convert issues data to DataFrame and perform analysis
    df = pd.DataFrame(issues_data)
    grouped_data = df.groupby(['closed_at'], as_index=False).count()
    time_series_data = grouped_data[['closed_at', 'issue_number']]
    time_series_data.columns = ['ds', 'y']

    # Decompose the time series data and plot observations
    decomposition = sm.tsa.seasonal_decompose(time_series_data.set_index('ds')['y'], period=len(time_series_data)//2)
    observation_plot = decomposition.plot()
    observation_plot.set_size_inches(12, 7)
    plt.title("Observations plot of closed issues")
    observation_plot.get_figure().savefig(local_image_path + observation_image_name)

    # Fit an ARIMA model and plot the forecast
    model = sm.tsa.ARIMA(time_series_data['y'].iloc[1:], order=(1, 0, 0))
    model_results = model.fit()
    time_series_data['forecast'] = model_results.fittedvalues
    forecast_plot = time_series_data[['y', 'forecast']].plot(figsize=(12, 7))
    plt.title("Timeseries forecasting of closed issues")
    forecast_plot.get_figure().savefig(local_image_path + forecast_image_name)

    # Construct and return the JSON response with image URLs
    response_data = {
        "observation_image_url": base_image_path + observation_image_name,
        "forecast_image_url": base_image_path + forecast_image_name
    }
    return jsonify(response_data)


@app.route('/api/statmcommits', methods=['POST'])
def analyze_commit_data():
    """
    Handles POST request to analyze commit data.
    Generates observation and forecast plots and uploads them to Google Cloud Storage.
    """

    # Extract commit data from the request body
    request_data = request.get_json()
    commit_data = request_data["pull"]
    repository_name = request_data["repo"]
    analysis_type = request_data["type"]

    # Environment variables setup
    base_image_path = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path')
    bucket_name = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')
    local_image_path = "static/images/"

    # File names for the images
    observation_image_name = f"stats_observation_{analysis_type}_{repository_name}.png"
    forecast_image_name = f"stats_forecast_{analysis_type}_{repository_name}.png"

    # Data processing and time series analysis
    df = pd.DataFrame(commit_data)
    grouped_data = df.groupby(['created_at'], as_index=False).count()
    time_series_data = grouped_data[['created_at', 'commit_number']]
    time_series_data.columns = ['ds', 'y']

    # Observation plot generation
    decomposition = sm.tsa.seasonal_decompose(time_series_data.set_index('ds')['y'], period=len(time_series_data)//2)
    observation_plot = decomposition.plot()
    observation_plot.set_size_inches(12, 7)
    plt.title("Observations plot of commits")
    observation_plot.get_figure().savefig(local_image_path + observation_image_name)

    # Forecast plot generation
    model = sm.tsa.ARIMA(time_series_data['y'].iloc[1:], order=(1, 0, 0))
    model_results = model.fit()
    time_series_data['forecast'] = model_results.fittedvalues
    forecast_plot = time_series_data[['y', 'forecast']].plot(figsize=(12, 7))
    plt.title("Timeseries forecasting of commits")
    forecast_plot.get_figure().savefig(local_image_path + forecast_image_name)

    # Initialize Google Cloud Storage client and upload the images
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Upload observation image
    observation_blob = bucket.blob(observation_image_name)
    observation_blob.upload_from_filename(local_image_path + observation_image_name)

    # Upload forecast image
    forecast_blob = bucket.blob(forecast_image_name)
    forecast_blob.upload_from_filename(local_image_path + forecast_image_name)

    # Construct and return the response with image URLs
    response_data = {
        "stats_observation_url": base_image_path + observation_image_name,
        "stats_forecast_url": base_image_path + forecast_image_name
    }
    return jsonify(response_data)

@app.route('/api/fbprophetis', methods=['POST'])
def forecast_issues_with_fbprophet():
    """
    Handles POST request to forecast GitHub issues data using Facebook Prophet.
    Generates and uploads forecast and component plots to Google Cloud Storage.
    """

    # Extract data from the request body
    request_data = request.get_json()
    analysis_type = request_data["type"]
    repository_name = request_data["repo"]
    issues_data = request_data["issues"]

    # Set up environment variables for image paths
    base_image_path = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path')
    bucket_name = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')
    local_image_path = "static/images/"

    # Filenames for the forecast and components images
    forecast_image_name = f"fbprophet_forecast_{analysis_type}_{repository_name}.png"
    forecast_components_image_name = f"fbprophet_forecast_components_{analysis_type}_{repository_name}.png"

    # Convert issues data to DataFrame and prepare for forecasting
    df = pd.DataFrame(issues_data)
    grouped_data = df.groupby(['created_at'], as_index=False).count()
    forecast_data = grouped_data[['created_at', 'issue_number']]
    forecast_data.columns = ['ds', 'y']

    # Fit the Prophet model and make predictions
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(forecast_data)
    future_data = prophet_model.make_future_dataframe(periods=60)
    forecast = prophet_model.predict(future_data)

    # Generate and save forecast and components plots
    forecast_plot = prophet_model.plot(forecast)
    components_plot = prophet_model.plot_components(forecast)
    forecast_plot.savefig(local_image_path + forecast_image_name)
    components_plot.savefig(local_image_path + forecast_components_image_name)

    # Initialize Google Cloud Storage client and upload the images
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Upload forecast image
    forecast_blob = bucket.blob(forecast_image_name)
    forecast_blob.upload_from_filename(local_image_path + forecast_image_name)

    # Upload forecast components image
    components_blob = bucket.blob(forecast_components_image_name)
    components_blob.upload_from_filename(local_image_path + forecast_components_image_name)

    # Construct and return the response with image URLs
    response_data = {
        "fbprophet_forecast_url": base_image_path + forecast_image_name,
        "fbprophet_forecast_components_url": base_image_path + forecast_components_image_name
    }
    return jsonify(response_data)


@app.route('/api/fbprophetis', methods=['POST'])
def forecast_issues_with_fbprophet():
    """
    Handles POST request to forecast GitHub issues data using Facebook Prophet.
    Generates and uploads forecast and component plots to Google Cloud Storage.
    """

    # Extract data from the request body
    request_data = request.get_json()
    analysis_type = request_data["type"]
    repository_name = request_data["repo"]
    issues_data = request_data["issues"]

    # Set up environment variables for image paths
    base_image_path = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path')
    bucket_name = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')
    local_image_path = "static/images/"

    # Filenames for the forecast and components images
    forecast_image_name = f"fbprophet_forecast_{analysis_type}_{repository_name}.png"
    forecast_components_image_name = f"fbprophet_forecast_components_{analysis_type}_{repository_name}.png"

    # Convert issues data to DataFrame and prepare for forecasting
    df = pd.DataFrame(issues_data)
    grouped_data = df.groupby(['created_at'], as_index=False).count()
    forecast_data = grouped_data[['created_at', 'issue_number']]
    forecast_data.columns = ['ds', 'y']

    # Fit the Prophet model and make predictions
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(forecast_data)
    future_data = prophet_model.make_future_dataframe(periods=60)
    forecast = prophet_model.predict(future_data)

    # Generate and save forecast and components plots
    forecast_plot = prophet_model.plot(forecast)
    components_plot = prophet_model.plot_components(forecast)
    forecast_plot.savefig(local_image_path + forecast_image_name)
    components_plot.savefig(local_image_path + forecast_components_image_name)

    # Initialize Google Cloud Storage client and upload the images
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Upload forecast image
    forecast_blob = bucket.blob(forecast_image_name)
    forecast_blob.upload_from_filename(local_image_path + forecast_image_name)

    # Upload forecast components image
    components_blob = bucket.blob(forecast_components_image_name)
    components_blob.upload_from_filename(local_image_path + forecast_components_image_name)

    # Construct and return the response with image URLs
    response_data = {
        "fbprophet_forecast_url": base_image_path + forecast_image_name,
        "fbprophet_forecast_components_url": base_image_path + forecast_components_image_name
    }
    return jsonify(response_data)

@app.route('/api/fbprophetcommits', methods=['POST'])
def fbprophetcommits():
    body = request.get_json()
    commit_response = body["pull"]
    repo_name = body["repo"]
    type = body["type"]
    print("type:",type)

    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'Your_Base_Image_path')

    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'Your_BUCKET_NAME')

    LOCAL_IMAGE_PATH = "static/images/"
    FORECAST_IMAGE_NAME = "fbprophet_forecast_" + type +"_"+ repo_name + ".png"
    FORECAST_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME

    FORECAST_COMPONENTS_IMAGE_NAME = "fbprophet_forecast_components_" + type +"_" + repo_name + ".png"
    FORECAST_COMPONENTS_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME

    df = pd.DataFrame(commit_response)
    df1 = df.groupby(['created_at'], as_index = False).count()
    dataFrame = df1[['created_at','commit_number']]
    dataFrame.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True)
    model.fit(dataFrame)
    future = model.make_future_dataframe(periods=60)
    forecast = model.predict(future)
    forcast_fig1 = model.plot(forecast)
    forcast_fig2 = model.plot_components(forecast)
    forcast_fig1.savefig(LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)
    forcast_fig2.savefig(LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)

    

    # Uploads an images into the google cloud storage bucket
    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)

    new_blob = bucket.blob(FORECAST_COMPONENTS_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)

    # Construct the response
    json_response = {
        "fbprophet_forecast_url": FORECAST_IMAGE_URL,
        "fbprophet_forecast_components_url": FORECAST_COMPONENTS_IMAGE_URL
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)

@app.route('/api/fbprophetis', methods=['POST'])
def forecast_issues_with_fbprophet():
    """
    Handles POST request to forecast GitHub issues data using Facebook Prophet.
    Generates and uploads forecast and component plots to Google Cloud Storage.
    """

    # Extract data from the request body
    request_data = request.get_json()
    analysis_type = request_data["type"]
    repository_name = request_data["repo"]
    issues_data = request_data["issues"]

    # Set up environment variables for image paths
    base_image_path = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path')
    bucket_name = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')
    local_image_path = "static/images/"

    # Filenames for the forecast and components images
    forecast_image_name = f"fbprophet_forecast_{analysis_type}_{repository_name}.png"
    forecast_components_image_name = f"fbprophet_forecast_components_{analysis_type}_{repository_name}.png"

    # Convert issues data to DataFrame and prepare for forecasting
    df = pd.DataFrame(issues_data)
    grouped_data = df.groupby(['created_at'], as_index=False).count()
    forecast_data = grouped_data[['created_at', 'issue_number']]
    forecast_data.columns = ['ds', 'y']

    # Fit the Prophet model and make predictions
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(forecast_data)
    future_data = prophet_model.make_future_dataframe(periods=60)
    forecast = prophet_model.predict(future_data)

    # Generate and save forecast and components plots
    forecast_plot = prophet_model.plot(forecast)
    components_plot = prophet_model.plot_components(forecast)
    forecast_plot.savefig(local_image_path + forecast_image_name)
    components_plot.savefig(local_image_path + forecast_components_image_name)

    # Initialize Google Cloud Storage client and upload the images
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Upload forecast image
    forecast_blob = bucket.blob(forecast_image_name)
    forecast_blob.upload_from_filename(local_image_path + forecast_image_name)

    # Upload forecast components image
    components_blob = bucket.blob(forecast_components_image_name)
    components_blob.upload_from_filename(local_image_path + forecast_components_image_name)

    # Construct and return the response with image URLs
    response_data = {
        "fbprophet_forecast_url": base_image_path + forecast_image_name,
        "fbprophet_forecast_components_url": base_image_path + forecast_components_image_name
    }
    return jsonify(response_data)


@app.route('/api/commits', methods=['POST'])
def analyze_commits():
    """
    Processes commit data and applies LSTM for forecasting.
    Generates plots and uploads them to Google Cloud Storage.
    """

    # Extract data from the request body
    request_data = request.get_json()
    commit_data = request_data["pull"]
    repository_name = request_data["repo"]
    analysis_type = request_data["type"]

    # Prepare data for LSTM
    df = pd.DataFrame(commit_data)
    df_grouped = df.groupby(["created_at"], as_index=False).count()
    df_prepared = df_grouped[["created_at", 'commit_number']]
    df_prepared.columns = ['ds', 'y']
    df_prepared['ds'] = pd.to_datetime(df_prepared['ds'])

    # Normalize data and prepare training and test sets
    scaler = MinMaxScaler(feature_range=(0, 1))
    commit_values = scaler.fit_transform(df_prepared['y'].values.reshape(-1, 1))

    # Split data into training and testing
    train_size = int(len(commit_values) * 0.80)
    test_size = len(commit_values) - train_size
    train, test = commit_values[0:train_size], commit_values[train_size:len(commit_values)]

    # Define dataset creation function for LSTM
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = min(30, len(test) - 2)
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back) if len(test) > look_back + 1 else (None, None)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(100, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    history = model.fit(X_train, np.reshape(Y_train, (Y_train.shape[0], 1)), epochs=20, batch_size=70, validation_data=(X_test, np.reshape(Y_test, (Y_test.shape[0], 1))),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

    # Generate and save plots
    def generate_and_save_plot(image_path, title):
        plt.figure(figsize=(8, 4))
        plt.title(title)
        plt.ylabel('Value')
        plt.xlabel('Time')
        plt.legend(loc='upper right')
        plt.savefig(image_path)

    base_image_path = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path')
    bucket_name = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')
    local_image_path = "static/images/"

    model_loss_image_name = f"model_loss_{analysis_type}_{repository_name}.png"
    generate_and_save_plot(local_image_path + model_loss_image_name, 'Model Loss')

    lstm_generated_image_name = f"lstm_generated_data_{analysis_type}_{repository_name}.png"
    generate_and_save_plot(local_image_path + lstm_generated_image_name, 'LSTM Generated Data')

    all_issues_data_image_name = f"all_issues_data_{analysis_type}_{repository_name}.png"
    generate_and_save_plot(local_image_path + all_issues_data_image_name, 'All Issues Data')

    # Upload images to Google Cloud Storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    def upload_to_gcs(image_name):
        blob = bucket.blob(image_name)
        blob.upload_from_filename(local_image_path + image_name)

    upload_to_gcs(model_loss_image_name)
    upload_to_gcs(lstm_generated_image_name)
    upload_to_gcs(all_issues_data_image_name)

    # Construct the response with URLs
    json_response = {
        "model_loss_image_url": base_image_path + model_loss_image_name,
    }


@app.route('/api/pulls', methods=['POST'])
def analyze_pull_requests():
    """
    Analyzes pull request data using LSTM for forecasting.
    Generates and uploads plots for model loss, LSTM generated data, and all pull request data.
    """

    # Extract data from the request body
    request_data = request.get_json()
    pull_request_data = request_data["pull"]
    repository_name = request_data["repo"]
    analysis_type = request_data["type"]

    # Prepare data for LSTM
    df = pd.DataFrame(pull_request_data)
    df_grouped = df.groupby(["created_at"], as_index=False).count()
    df_prepared = df_grouped[["created_at", 'pull_req_number']]
    df_prepared.columns = ['ds', 'y']
    df_prepared['ds'] = pd.to_datetime(df_prepared['ds'])

    # Normalize data and prepare training and test sets
    scaler = MinMaxScaler(feature_range=(0, 1))
    pull_request_values = scaler.fit_transform(df_prepared['y'].values.reshape(-1, 1))

    # Split data into training and testing
    train_size = int(len(pull_request_values) * 0.80)
    test_size = len(pull_request_values) - train_size
    train, test = pull_request_values[0:train_size], pull_request_values[train_size:len(pull_request_values)]

    # Define dataset creation function for LSTM
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = min(30, len(test) - 2)
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back) if len(test) > look_back + 1 else (None, None)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(100, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    history = model.fit(X_train, np.reshape(Y_train, (Y_train.shape[0], 1)), epochs=20, batch_size=70, validation_data=(X_test, np.reshape(Y_test, (Y_test.shape[0], 1))),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

    # Generate and save plots
    base_image_path = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path')
    bucket_name = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')
    local_image_path = "static/images/"

    model_loss_image_name = f"model_loss_{analysis_type}_{repository_name}.png"
    lstm_generated_image_name = f"lstm_generated_data_{analysis_type}_{repository_name}.png"
    all_pulls_data_image_name = f"all_pulls_data_{analysis_type}_{repository_name}.png"

    # Save plots
    def save_plot(image_name, title, y_label, plot_data):
        plt.figure(figsize=(10, 4))
        plt.plot(plot_data)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel('Time')
        plt.savefig(local_image_path + image_name)

    save_plot(model_loss_image_name, 'Model Loss for Pull Requests', 'Loss', history.history['loss'])
    save_plot(lstm_generated_image_name, 'LSTM Generated Data for Pull Requests', 'Pull Requests', model.predict(X_test))
    save_plot(all_pulls_data_image_name, 'All Pull Request Data', 'Pull Requests', pull_request_values)

    # Upload images to Google Cloud Storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    def upload_to_gcs(image_name):
        blob = bucket.blob(image_name)


# Run LSTM app server on port 8080
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
