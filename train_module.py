import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler


class Trainer_Module(object):

    def __init__(self, uv_time_series, n_steps, n_features, val_split, factor,patience, learning_rate, min_delta, epochs, batch_size, monitor_parameter):

        self.univariate_time_series = uv_time_series
        self.n_steps                = n_steps
        self.n_features             = n_features
        self.val_split              = val_split
        self.patience               = patience
        self.learning_rate          = learning_rate
        self.min_delta              = min_delta
        self.epochs                 = epochs
        self.batch_size             = batch_size
        self.monitor_parameter      = monitor_parameter
    

    def _train(self):
        self._scale_data()
        self._reshape_data()
        self._build_and_fit_model()
    
    def _scale_data(self):
      scaler                      = StandardScaler()
      univariate_data             = self.univariate_time_series
      scaled_data                 = scaler.fit_transform(univariate_data.values.reshape(-1, 1))
      self.scaler                 =  scaler
      self.scaled_data            =  scaled_data
      print("SCALED DATA SUCCESSFULLY")

    def _reshape_data(self):
        
        univariate_data             = self.scaled_data
        n_steps                     = self.n_steps
        n_features                  = self.n_features

        X, y                        = [], []

        for i in range(len(univariate_data)):
            end_ix                  = i + self.n_steps
            if end_ix > len(univariate_data) - 1:
                break
            seq_x, seq_y            = univariate_data[i:end_ix], univariate_data[end_ix]
            X.append(seq_x)
            y.append(seq_y)

        X = np.array(X)
        y = np.array(y)
        print("X shape before: ", X.shape)
    
        X                           = X.reshape((X.shape[0], X.shape[1], n_features))
        self.X                      = X
        self.y                      = y
        print("X shape after: ", X.shape)


    def _build_and_fit_model(self):
        
        monitor                     = self.monitor_parameter
        patience                    = self.patience
        min_delta                   = self.min_delta
        n_steps                     = self.n_steps
        n_features                  = self.n_features
        epochs                      = self.epochs
        batch_size                  = self.batch_size
        learning_rate               = self.learning_rate
        val_split                   = self.val_split
        activation_function         = 'relu'
        verbose                     = 0
        mode                        = 'min'
        restore_best_weights        = True
        optimizer                   = 'adam'
        loss                        = 'mse'
        early_stopping              = EarlyStopping(monitor=monitor, patience=patience, min_delta=0, verbose=verbose, mode=mode, restore_best_weights=restore_best_weights)
        self.early_stopping         = early_stopping

        X                           = self.X
        y                           = self.y
        print("Xshape before fitting", X.shape)
        print("yshape before fitting", y.shape)
        model                       = Sequential()
        model                        .add(LSTM(64, activation=activation_function, return_sequences=True, input_shape=(n_steps, n_features)))
        model                        .add(LSTM(32, activation=activation_function, return_sequences=False))
        model                        .add(Dropout(0.2))
        model                        .add(Dense(1))

        model                        .compile(optimizer=optimizer, loss=loss)
        K                            .set_value(model.optimizer.learning_rate, learning_rate)

        history                     = model.fit(X, y, epochs=epochs, callbacks=[early_stopping],batch_size=batch_size, validation_split=val_split, verbose=verbose)

        self.model                  = model
        self.history                = history
        
