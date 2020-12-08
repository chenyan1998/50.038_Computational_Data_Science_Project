import pandas as pd
import DataProcessor as dp
from sklearn import tree
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler

# data preparation
data_df = pd.read_csv('./newdata/2019.csv')

# visualise how balance our dataset is
sns.countplot(x="Hit", data=data_df, palette="muted")

X_train, X_test, y_train, y_test = train_test_split(data_df.drop(['Hit'],axis=1), 
                                                    data_df['Hit'],
                                                    test_size=0.3,
                                                    random_state=0) 
# define a new scaler: 
x_scaler = MinMaxScaler() 
x_scaler.fit(X_train)
X_train_norm = x_scaler.transform(X_train)
X_test_norm = x_scaler.transform(X_test)

#%% training
DT = tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=2) 
DT.fit(X_train,y_train) # Using default parameters
plt.figure(figsize=(14,12))
tree.plot_tree(DT)
train_score = DT.score(X_train,y_train)


#%%Predicting labels for our test set using model
# evalutate and visualise the result
y_pred = DT.predict(X_test_norm) 
report,rocfig=dp.evaluate_on_training_set(y_test, y_pred) 
pred_fig=dp.plot_pred_original(y_pred,y_test,'Decision Tree')
report['training score']=train_score

#%%
# save predict result
rocfig.savefig('./result/DecisonTree/DTroc-10.png')
print(report)
with open('./result/DecisonTree/DTreport-10', 'w') as f:
    [f.write('{0}:\n{1}\n'.format(key, value)) for key, value in report.items()]
pred_fig.savefig('./result/DecisonTree/DTpred-10.png')

#%%
joblib.dump(DT, './trainedModel/DT-10.sav')

# load model
# DT=joblib.load('./DT.sav)