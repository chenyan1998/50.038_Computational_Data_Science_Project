from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import DataProcessor as dp
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np

# data preparation
data_df = pd.read_csv('./newdata/2019.csv')

# visualise how balance our dataset is
# sns.countplot(x="Hit", data=data_df, palette="muted")

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
acc =[]
for k in range(1,150): # try out different k values,found 38~41 is approprite 
    KN = KNeighborsClassifier(n_neighbors=k)
    KN.fit(X_train_norm, y_train) # Training the model
    # Predicting labels and evaluate
    y_pred = KN.predict(X_test_norm)
    acc.append(accuracy_score(y_test,y_pred))
print(acc)

plt.plot(acc)
plt.title('Validation Accuracy')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.show()