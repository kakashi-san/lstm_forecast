import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Data_Handler(object):

        def __init__(self, path_to_server_data_dir, server_name, sampling_rate, n_components):

            self.path_to_data = path_to_server_data_dir
            self.server_name          = server_name
            self.columns              = []
            self.sampling_rate        = sampling_rate
            self.scaler               = None
            self.pca_n_components     = n_components
            self.pca                  = None



        def _process_source_data(self):

            self._load_data()
            self._process_data()

        
        def _load_data(self):

            path_to_data              = self.path_to_data
            data                      = pd.read_csv(path_to_data)
            data.index                = pd.to_datetime(data['Date'])
            data                      = data.drop(["Date"], axis=1).dropna()
            
            self.columns              = data.columns
            self.resampled_data       = data.resample(self.sampling_rate)
            self.start_time           = str(data.index[0])
            self.end_time             = str(data.index[-1])


        def _process_data(self):
            
            data                      = self.resampled_data
            data                      = data.interpolate()

            scaler                    = StandardScaler()
            scaled_array              = scaler.fit_transform(data)
            scaled_df                 = pd.DataFrame(scaled_array, columns=data.columns)
            self.scaled_data          = scaled_df
            self.scaler               = scaler

 
            n_components              = self.pca_n_components
            pca                       = PCA(n_components=n_components)
            pca_df                    = pca.fit_transform(scaled_df)
            pca_df                    = pd.DataFrame(pca_df)
            pca_df.index              = data.index
            self.pca_data             = pca_df
            self.pca                  = pca