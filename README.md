# 50.038 Computational Data Science Project
## Hit Song Prediction
When we listen to songs, despite the different taste in music, one can more or less deduce whether a song will be popular. However, rather than using our intuition or "gut-feeling" to predict hit songs, the purpose of the project is to see if we can use intrinsic music data to identify Hit songs by machine learning.


# 



## Group 8 
- Chenyan       1003620
- Dongke        1003713
- Gladys Chua   1003585
- Pang Luying   1003631

#

## Get your spotify credentials
Since we are using spotify API to extract the features of songs, we need spotify credentials for the code to run properly. You can follow the steps below to get your own spotify credentials.
- Login your spotify developer account or create a spotify developer account if you do not have one.
- Create a client ID, you can do by pressing the green CREATE A CLIENT ID button.
- You will get a pop up. On the pop up, give it a any name you want, choose non-commercial use, agree all the terms and conditions and submit the from.
- Go to your developer dashboard page, click on the new app you just created.
- On your app’s dashboard page, you’ll see your client ID on the top left-hand side. Copy paste this client ID to the right place in DataGen.py and projectUI.py file.
- Underneath your client ID, you’ll see “Show Client Secret” in green. Click on it and you will see your Client Secret. Copy paste this client secret to the right place in DataGen.py and projectUI.py file.
- Done.

#

## UI demo
```
pip install streamlit
```

### To run the code
At the command line: 
``` 
streamlit run projectUI.py  
```

### How to use
1. Input the artist and song 
2. Check to get song from Spotify
3. Click on the button to predict if the song is a Hit song

