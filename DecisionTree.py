import pandas as pd
import DataProcessor as dp
from sklearn import tree
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

# data preparation
data_df = pd.read_csv('./CleanData/01.csv')

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
DT = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=2) 
DT.fit(X_train,y_train) # Using default parameters
plt.figure(figsize=(14,12))
tree.plot_tree(DT)


#%%Predicting labels for our test set using model
# evalutate and visualise the result
y_pred = DT.predict(X_test) 
# print (y_pred)
dp.evaluate_on_training_set(y_test, y_pred) 
pred_fig=dp.plot_pred_original(y_pred,y_test,'Decision Tree')
pred_fig.savefig('./pred_fig/Logistic Regression')

joblib.dump(DT, './DT.sav')

# load model
# DT=joblib.load('./DT.sav)