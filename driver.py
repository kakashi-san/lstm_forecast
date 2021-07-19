import pandas as pd
import numpy as np
import os
from data_handler import Data_Handler
from train_module import Trainer_Module
from prediction_module import Predicter_Module


sampling_rate           = '24H'
n_steps                 = 7
n_features              = 1
n_components            = 3
val_split               = 0.80
factor                  = 0.001
patience                = 40
learning_rate           = 0.001
min_delta               = 0.1
epochs                  = 100
batch_size              = 16

class Driver(object):

    def __init__(self, path_to_data, server_space, server_name):
        self.path_to_data            = path_to_data
        self.server_space            = server_space
        self.server_name             = server_name
        self.columns                 = None
        self.sampling_rate           = '24H'
        self.n_steps                 = 7
        self.n_features              = 1
        self.n_components            = 3
        self.val_split               = 0.80
        self.factor                  = 0.001
        self.patience                = 40
        self.learning_rate           = 0.001
        self.min_delta               = 0.1
        self.epochs                  = 100
        self.batch_size              = 16
        self.monitor_parameter       = 'val_loss'
        self.pca_data                = None
        self.start_time              = None
        self.end_time                = None
        self.pca_models              = []
        self.pca_future              = []
        self.pca_future_upper        = []
        self.pca_future_lower        = []
        self.predictions             = []
        self.predictions_upper       = []
        self.predictions_lower       = []

    def _drive(self):
        self._handle_data()
        self._loops_over_pca_components_trains_and_predicts()
        self._inverse_scale_transform()

    def _handle_data(self):
        dh                           = Data_Handler(
            path_to_server_data_dir  = self.path_to_data,
            server_name              = self.server_name,
            sampling_rate            = self.sampling_rate,
            n_components             = self.n_components)
             
        dh._process_source_data()

        self.pca_data                 = dh.pca_data
        self.start_time               = dh.start_time
        self.end_time                 = dh.end_time
        self.global_scaler            = dh.scaler
        self.pca                      = dh.pca
        self.columns                  = dh.columns

    def _loops_over_pca_components_trains_and_predicts(self):

        for i_component in range(self.n_components):
            pca_component              = self.pca_data.iloc[:, i_component]

            train_module               = Trainer_Module(
                     uv_time_series    = pca_component, 
                     n_steps           = self.n_steps, 
                     n_features        = self.n_features,
                     val_split         = self.val_split,
                     factor            = self.factor,
                     patience          = self.patience,
                     learning_rate     = self.learning_rate,
                     min_delta         = self.min_delta,
                     epochs            = self.epochs,
                     batch_size        = self.batch_size,
                     monitor_parameter = self.monitor_parameter)
            
            train_module              ._train()
            trained_model             = train_module.model
            self.pca_models           .append(trained_model) 
            predict_start_time        = self.end_time
            scaler                    = train_module.scaler

            prediction_module         = Predicter_Module(model=trained_model, 
                      uv_time_series  = pca_component,
                      n_steps         = self.n_steps, 
                      n_features      = self.n_features,
                      future_period   = 45, 
                      start_date      = predict_start_time,
                      frequency       = '24H',
                      scaler          = scaler)
            
            prediction_module         ._make_predictions()
            
            self.pca_future           .append(prediction_module.future_values)
            self.pca_future_upper     .append(prediction_module.future_values_upper)
            self.pca_future_lower     .append(prediction_module.future_values_upper)
            self.future_time          = prediction_module.future_time
    
    def _inverse_scale_transform(self):
        self.predictions              = pd.DataFrame(self.global_scaler.inverse_transform(self.pca.inverse_transform(np.array(self.pca_future).T))).reset_index(drop=True)
        self.predictions_upper        = pd.DataFrame(self.global_scaler.inverse_transform(self.pca.inverse_transform(np.array(self.pca_future_upper).T))).reset_index(drop=True)
        self.predictions_lower        = pd.DataFrame(self.global_scaler.inverse_transform(self.pca.inverse_transform(np.array(self.pca_future_lower).T))).reset_index(drop=True)

    
    def _format_pred_data_for_frontend(self):
      self.predictions.columns        = self.columns
      self.predictions_upper.columns  = self.columns
      self.predictions_lower.columns  = self.columns

      pred                            = self.predictions
      fore_date                       = self.future_time
      lower_pred                      = self.predictions_lower
      upper_pred                      = self.predictions_upper
      predict_dict = {}
      for col in self.columns:
        predict_dict[col]               = {'yhat': self._endpoint_format( pred[col].tolist(),fore_date),
                    'yhat_lower'        : self._endpoint_format( lower_pred[col].tolist(),fore_date),
                    'yhat_upper'        : self._endpoint_format( upper_pred[col].tolist(), fore_date)
                    }
      self.predict_dict = predict_dict
    
    def _endpoint_format(self, A, B):
      return [[B[ix], A[ix]] for ix in range(len(A))]


