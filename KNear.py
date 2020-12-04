from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import DataProcessor as dp
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
# normalise dataset
x_scaler = MinMaxScaler() 
x_scaler.fit(X_train)
X_train_norm = x_scaler.transform(X_train)
X_test_norm = x_scaler.transform(X_test)

#%% Define and train the model
KN = KNeighborsClassifier(n_neighbors=5)
KN.fit(X_train_norm, y_train) # Training the model
#%% Predicting labels and evaluate
y_pred = KN.predict(X_test_norm) 
report,roc=dp.evaluate_on_training_set(y_test, y_pred) 
pred_fig=dp.plot_pred_original(y_pred,y_test,'K Nearest Neighbor')
pred_fig.savefig('./pred_fig/K Nearest Neighbor')

#%%
joblib.dump(KN, './KN.sav')