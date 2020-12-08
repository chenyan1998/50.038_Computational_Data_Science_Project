from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import DataProcessor as dp
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

# data preparation
data_df = pd.read_csv('./newdata/2019.csv')
data_df=data_df.drop(['danceability','energy','loudness','acousticness','instrumentalness','liveness','tempo','time_signature'],axis=1)
# visualise how balance our dataset is
sns.countplot(x="Hit", data=data_df, palette="muted")

X_train, X_test, y_train, y_test = train_test_split(data_df.drop(['Hit'],axis=1), 
                                                    data_df['Hit'],
                                                    test_size=0.3,
                                                    random_state=0) 
# normalise dataset
x_scaler = MinMaxScaler() 
x_scaler.fit(X_train)
X_train_norm = x_scaler.transform(X_train)
X_test_norm = x_scaler.transform(X_test)

#%% Define and train the model
k=39
KN = KNeighborsClassifier(n_neighbors=k)
KN.fit(X_train_norm, y_train) # Training the model
train_score = KN.score(X_train_norm,y_train)
# perform permutation importance
results = permutation_importance(KN, X_train_norm, y_train,random_state=0)
feature_importance = results.importances_mean
print(feature_importance)
#%%
# Predicting labels and evaluate
y_pred = KN.predict(X_test_norm)
report,rocfig =dp.evaluate_on_training_set(y_test, y_pred) 
pred_fig=dp.plot_pred_original(y_pred,y_test,'K Nearest Neighbor')
report['training score']=train_score

#%%save predict result
features=list(data_df.columns)
for i,v in enumerate(feature_importance):
    report[features[i]]=v
rocfig.savefig('./result/K Nearest Neighbor/KNNLroc'+str(k)+'.png')

print(report)
with open('./result/K Nearest Neighbor/KNNLreport'+str(k), 'w') as f:
    [f.write('{0}:\n{1}\n'.format(key, value)) for key, value in report.items()]
pred_fig.savefig('./result/K Nearest Neighbor/KNNLpred'+str(k)+'.png')

#%%
joblib.dump(KN, './trainedModel/KNL'+str(k)+'.sav')