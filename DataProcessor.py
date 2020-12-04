
# label top 15 as hit songs, bottom 15 as non-hit songs
import numpy as np
def label_songs(df):
    hit = np.ones(shape=(15,1))
    nonhit = np.zeros(shape=(15,1))
    label = np.concatenate((hit,nonhit))

    df["Hit"] = label

    return df

# Extract audio features
import pandas as pd
def features(sp,df):
    result = df.apply(lambda row: sp.audio_features(row.URL),axis=1)
    feature_names= ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness',                           'valence','tempo','type','id','uri','track_href','analysis_url','duration_ms','time_signature']
    drop_features=['type','id','uri','track_href','analysis_url']
    features_df = pd.DataFrame(columns=feature_names)
    for row in result:
        features_df = features_df.append(row[0],ignore_index=True)
    
    features_df = features_df.drop(drop_features,axis=1)
    return features_df


def categorize(df):
    for key in ['danceability','energy','key','loudness','mode','speechiness', 'acousticness','instrumentalness','liveness','valence','tempo', 'duration_ms','time_signature','Hit']:
        df[key] = pd.Categorical(df[key])
        df[key] = df[key].cat.codes
    return df


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
def evaluate_on_training_set(y_test, y_pred):
  result={}
  # Calculate AUC
  result['AUC']=roc_auc_score(y_test,y_pred)
  print("AUC is: ", roc_auc_score(y_test,y_pred) )
  # recall and precision
  result['report']=classification_report(y_test, y_pred)
  print(classification_report(y_test, y_pred))
  # confusion matrix
  result['ConMatrix']=confusion_matrix(y_test, y_pred)
  print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

  # calculate points for ROC curve
  fpr, tpr, thresholds = roc_curve(y_test, y_pred)
  # Plot ROC curve
  fig, ax = plt.subplots()
  ax.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_score(y_test, y_pred))
  ax.plot([0, 1], [0, 1], 'k--')  # random predictions curve
  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.0])
  ax.set_xlabel('False Positive Rate or (1 - Specifity)')
  ax.set_ylabel('True Positive Rate or (Sensitivity)')
  ax.set_title('Receiver Operating Characteristic')
  
  
  return result,fig


def plot_pred_original(y_pred,y_test,model_type):
    fig,ax = plt.subplots(figsize=(20,3))
    ax.scatter(np.arange(len(y_test)),y_test,c='b',alpha=0.5,label='Original data')
    ax.scatter(np.arange(len(y_pred)),y_pred,c='r',alpha=0.5,label='Predicted data')
    ax.legend()
    ax.set_ylabel('Target')
    ax.set_xlabel('Songs')
    ax.set_title(model_type)
    return fig