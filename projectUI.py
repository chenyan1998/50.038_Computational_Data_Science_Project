import streamlit as st
import pandas as pd
import pickle
import numpy as np
import spotipy
import joblib
import plotly.express as px
from spotipy.oauth2 import SpotifyClientCredentials     # to access authorised Spotify data

# Add in your client_id and client_secret inside ''
client_id = ''
client_secret = ''

client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)        # spotify object to access API

model = joblib.load('./trainedModel/KN39.sav')
scaler = joblib.load('./trainedModel/knn_scaler.sav')
rdmodel = joblib.load("./trainedModel/RF.sav")
admodel = joblib.load("./trainedModel/Ada.sav")


def getSong(artist,song):
    q="artist:%@"+artist+" track:%@"+song
    # print("inside function",q,type(song))
    spotify_url=''
    options={}
    result = sp.search(q) #search query
    for i,t in enumerate(result['tracks']['items']):
        print(i,t['name'],t['external_urls'])
        options[t['name']]=t['external_urls']['spotify']
    return options

def predictSong(option):
    spotify_url = option
    features = sp.audio_features(spotify_url)
    feature_names= ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','type','id','uri','track_href','analysis_url','duration_ms','time_signature']
    drop_features=['type','id','uri','track_href','analysis_url']
    features_df = pd.DataFrame(columns=feature_names)
    features_df = features_df.append(features[0],ignore_index=True)
    features_df = features_df.drop(drop_features,axis=1)
    test_norm = scaler.transform(features_df.to_numpy())
    y_pred=model.predict(test_norm)
    rf_pred = rdmodel.predict(test_norm)
    ad_pred = admodel.predict(test_norm)
    return int(y_pred),int(rf_pred),int(ad_pred),spotify_url

def main():
    st.sidebar.title('Hit Song Prediction')
    st.sidebar.markdown('CDS Project Group 8')
    tracks_df = pd.read_csv("./newdata/2019.csv")
    df_2020 = pd.read_csv("./newdata/2019.csv")
    if st.sidebar.checkbox("Data Visualisation"):
        st.title("plots")
        figu = px.scatter(tracks_df,x='tempo',y='energy',color='Hit',hover_data=['danceability','instrumentalness'],title='2019 tempo by energy')
        fig = px.scatter(df_2020,x='tempo',y='energy',color='Hit',hover_data=['danceability','instrumentalness'],title="2020 tempo by energy")
        st.plotly_chart(figu)
        st.plotly_chart(fig)
        st.vega_lite_chart(tracks_df, {'mark': {'type': 'circle', 'tooltip': True},
            'encoding': {
                'x': {'field': 'energy', 'type': 'quantitative'},         
                'y': {'field': 'valence', 'type': 'quantitative'},        
                'size': {'field': 'loudness', 'type': 'quantitative'},        
                'color': {'field': 'Hit', 'type': 'quantitative'},
                },
            }, use_container_width=True)
    if st.checkbox('Show Dataframe'):
        tracks_df
        df_2020
        # st.line_chart(tracks_df)

    artist = st.text_input("Artist Name:")
    song = st.text_input("Name of the song:")
    
    if st.checkbox("Get Song from Spotify"):
        options = getSong(artist,song)
        
        if len(options)==0:
            st.title("The song is not found/available in spotify. Do you want to check your spelling or try another song?")
        else:
            option = st.selectbox('which song you wanna choose',list(options.keys()))
            st.write('selected:', option)
            if st.button("Predict hit song"):
                output,output2,output3,url = predictSong(options[option])
                print(type(artist),song,output)
                if output == 1:
                    st.markdown("From the **KNN** model,\nThis song is predicted to be a **hit song** :sunglasses:")
                elif output == 0:
                    st.markdown("From the **KNN** model,\nThis song is predicted to be a **non-hit song**")
                if output2 == 1:
                    st.markdown("From the **Random Forest** model,\nThis song is predicted to be a **hit song** :sunglasses:")
                elif output2 == 0:
                    st.markdown("From the **Random Forest** model,\nThis song is predicted to be a **non-hit** song")
                if output3==1:
                    st.markdown("From the **Adaboost** model,\nThis song is predicted to be a **hit song** :sunglasses:")
                elif output3 == 0:
                    st.markdown("From the **Adaboost** model,\nThis song is predicted to be a **non-hit song**")
                st.markdown(url)

if __name__=="__main__":
    main()