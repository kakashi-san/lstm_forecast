from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend as K
from numpy import array
import pandas as pd
import numpy as np

class Predicter_Module():


    def __init__(self, model, uv_time_series, n_steps, n_features, future_period, start_date, frequency, scaler):

        self.univariate_time_series = uv_time_series
        self.model                  = model
        self.n_steps                = n_steps
        self.n_features             = n_features
        self.future_period          = future_period
        self.future_time            = []        
        self.future_values          = []
        self.future_values_upper    = []
        self.future_values_lower    = []
        self.start_date             = start_date
        self.frequency              = frequency
        self.scaler                 = scaler

    def _make_predictions(self):
        self._make_future_time()
        self._predict_future_univariate_data()
        self._predict_future_univariate_data_upper_limit()
        self._predict_future_univariate_data_lower_limit()


    def _make_future_time(self):
        start_date                  = self.start_date
        periods                     = self.future_period
        frequency                   = self.frequency

        t1                          = pd.date_range(start=start_date, periods=periods, freq=frequency)
        unix_time_float             = (t1 -pd.Timestamp("1970-01-01"))/ pd.Timedelta('1ms')
        unix_time_int               = [int(i) for i in unix_time_float]

        self.future_time            = unix_time_int

    

    def _predict_future_univariate_data(self):


 
        future_period               = self.future_period 
        model                       = self.model
        n_steps                     = self.n_steps
        n_features                  = self.n_features
        x_input                     = self.univariate_time_series.iloc[-1-n_steps:]
        temp_input                  = list(x_input)
        lst_output                  = []


        i = 0
        while i < future_period:

            if len(temp_input) > 5:
                x_input = array(temp_input[1:])

                x_input = x_input.reshape((1, n_steps, n_features))

                yhat = model.predict(x_input, verbose=0)

                temp_input.append(yhat[0][0])
                temp_input = temp_input[1:]

                lst_output.append(yhat[0][0])
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, n_features))
                yhat = model.predict(x_input, verbose=0)

                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
                i = i + 1

        self.future_values = self.scaler.inverse_transform( lst_output)


    def _predict_future_univariate_data_upper_limit(self):


        future_period               = self.future_period 
        model                       = self.model
        n_steps                     = self.n_steps
        n_features                  = self.n_features
        x_input                     = self.univariate_time_series.iloc[-1-n_steps:]
        temp_input                  = list(x_input)
        lst_output                  = []

        temp_input = list(x_input)
        lst_output = []
        i = 0
        while i < future_period:

            if len(temp_input) > 5:
                x_input = array(temp_input[1:])

                x_input = x_input.reshape((1, n_steps, n_features))

                yhat = model.predict(x_input, verbose=0)
                yhat = yhat + np.sqrt(abs(yhat))

                temp_input.append(yhat[0][0])
                temp_input = temp_input[1:]

                lst_output.append(yhat[0][0])
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, n_features))
                yhat = model.predict(x_input, verbose=0)
                yhat = yhat + np.sqrt(abs(yhat))

                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
                i = i + 1

        self.future_values_upper =  self.scaler.inverse_transform( lst_output)


    def _predict_future_univariate_data_lower_limit(self):

        future_period               = self.future_period 
        model                       = self.model
        n_steps                     = self.n_steps
        n_features                  = self.n_features
        x_input                     = self.univariate_time_series.iloc[-1-n_steps:]
        temp_input                  = list(x_input)
        lst_output                  = []

        temp_input = list(x_input)
        lst_output = []
        i = 0
        while i < future_period:

            if len(temp_input) > 5:
                x_input = array(temp_input[1:])

                x_input = x_input.reshape((1, n_steps, n_features))

                yhat = model.predict(x_input, verbose=0)
                yhat = yhat - np.sqrt(abs(yhat))

                temp_input.append(yhat[0][0])
                temp_input = temp_input[1:]

                lst_output.append(yhat[0][0])
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, n_features))
                yhat = model.predict(x_input, verbose=0)
                yhat = yhat - np.sqrt(abs(yhat))

                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
                i = i + 1

        self.future_values_lower= self.scaler.inverse_transform( lst_output)