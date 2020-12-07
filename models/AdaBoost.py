from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import DataProcessor as dp
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

data_df = pd.read_csv('./newdata/2019.csv')

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
Ada = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1) # Define the model with parameters
Ada.fit(X_train_norm, y_train) # Training the model
train_score = Ada.score(X_train,y_train)

#%% Predicting labels and evaluate
y_pred = Ada.predict(X_test_norm) 
report,rocfig=dp.evaluate_on_training_set(y_test, y_pred) 
pred_fig=dp.plot_pred_original(y_pred,y_test,'AdaBoost')
report['training score']=train_score

# save predict result
rocfig.savefig('./result/AdaBoost/AdaBoostroc.png')
print(report)
with open('./result/AdaBoost/AdaBoostreport', 'w') as f:
    [f.write('{0}:\n{1}\n'.format(key, value)) for key, value in report.items()]
pred_fig.savefig('./result/AdaBoost/AdaBoostpred.png')

#%%
joblib.dump(Ada, './trainedModel/Ada.sav')
