import os
import pandas as pd
import DataProcessor as dp
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Connect to API
client_id = 'PUT YOUR CLIENT ID'
client_secret = 'PUT YOUR CLIENT SECRET'

client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
#spotify object to access API
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 

directory = './spotifyChart'
input_df= None

for filename in os.listdir(directory):
    if '2020' in str(filename):
        data_df = pd.read_csv('./spotifyChart/'+filename,header=1)
        df=data_df.head(15).append(data_df.tail(15))
        features_df=dp.features(sp,df)
        tmp_df=dp.label_songs(features_df)
        input_df = pd.concat([input_df,tmp_df],ignore_index=True)
        
input_df.to_csv('./newData/2020.csv',index=False)