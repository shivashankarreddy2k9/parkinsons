import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#Read the data
from sklearn.metrics import f1_score
df=pd.read_csv('D:\\parkinsons-AI-master\\parkinsons.data')
df.head()


#Get the features and labels
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values
#Get the count of each label (0 and 1) in labels
(labels[labels==1].shape[0], labels[labels==0].shape[0])
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((-1, 1))
X = scaler.fit_transform(features)
Y = labels
###########################################################################

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=7)
# Train
#################################################################################
#############knn
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, Y)
y_pred_knn=neigh.predict(X_test);

accknn=accuracy_score (Y_test,y_pred_knn)
preknn=precision_score(Y_test,y_pred_knn,average='weighted')
recallknn=recall_score(Y_test,y_pred_knn,average='weighted')
fscoreknn=f1_score(Y_test,y_pred_knn,average='weighted')
cmknn=confusion_matrix(Y_test,y_pred_knn)
######################################################################
############logistic
from sklearn.linear_model import LogisticRegression
neigh = LogisticRegression(random_state=0)
neigh.fit(X, Y)
y_pred_log=neigh.predict(X_test);

acclog=accuracy_score (Y_test,y_pred_log)
prelog=precision_score(Y_test,y_pred_log,average='weighted')
recalllog=recall_score(Y_test,y_pred_log,average='weighted')
fscorelog=f1_score(Y_test,y_pred_log,average='weighted')
cmlog=confusion_matrix(Y_test,y_pred_log)
######################################################################
#######svm
from sklearn import svm
neigh = svm.SVC(decision_function_shape='ovo')
neigh.fit(X, Y)
y_pred_svm=neigh.predict(X_test);

accsvm=accuracy_score (Y_test,y_pred_svm)
presvm=precision_score(Y_test,y_pred_svm,average='weighted')
recallsvm=recall_score(Y_test,y_pred_svm,average='weighted')
fscoresvm=f1_score(Y_test,y_pred_svm,average='weighted')
msvm=confusion_matrix(Y_test,y_pred_svm)
####################################################################
#lsvm
from sklearn.svm import LinearSVC
neigh = LinearSVC(random_state=0, tol=1e-5)
neigh.fit(X, Y)
y_pred_lsvm=neigh.predict(X_test);

acclsvm=accuracy_score (Y_test,y_pred_lsvm)
prelsvm=precision_score(Y_test,y_pred_lsvm,average='weighted')
recalllsvm=recall_score(Y_test,y_pred_lsvm,average='weighted')
fscorelsvm=f1_score(Y_test,y_pred_lsvm,average='weighted')
cmlsvm=confusion_matrix(Y_test,y_pred_lsvm)
############################################################################
#naivebayes
from sklearn.naive_bayes import GaussianNB
neigh = GaussianNB()
neigh.fit(X, Y)
y_pred_nb=neigh.predict(X_test);

accnb=accuracy_score (Y_test,y_pred_nb)
prenb=precision_score(Y_test,y_pred_nb,average='weighted')
recallnb=recall_score(Y_test,y_pred_nb,average='weighted')
fscorenb=f1_score(Y_test,y_pred_nb,average='weighted')
cmnb=confusion_matrix(Y_test,y_pred_nb)
###########################################################################
######################decsion tree
from sklearn.tree import DecisionTreeClassifier
neigh = DecisionTreeClassifier(random_state=0, max_depth=2)
neigh.fit(X, Y)
y_pred_dt=neigh.predict(X_test);

accdt=accuracy_score (Y_test,y_pred_dt)
predt=precision_score(Y_test,y_pred_dt,average='weighted')
recalldt=recall_score(Y_test,y_pred_dt,average='weighted')
fscoredt=f1_score(Y_test,y_pred_dt,average='weighted')
cmknndt=confusion_matrix(Y_test,y_pred_dt)
##############################################################################
#########################random forest
from sklearn.ensemble import RandomForestClassifier
neigh = RandomForestClassifier(max_depth=2, random_state=0)
neigh.fit(X, Y)
y_pred_rf=neigh.predict(X_test);

accrf=accuracy_score (Y_test,y_pred_rf)
prerf=precision_score(Y_test,y_pred_rf,average='weighted')
recallrf=recall_score(Y_test,y_pred_rf,average='weighted')
fscorerf=f1_score(Y_test,y_pred_rf,average='weighted')
cmrf=confusion_matrix(Y_test,y_pred_rf)
#########################################
#############xgboost
import xgboost as xgb
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model.fit(X, Y)
y_pred_xg=neigh.predict(X_test);
model.score(X_test,Y_test)
accxg=accuracy_score (Y_test,y_pred_xg)
prexg=precision_score(Y_test,y_pred_xg,average='weighted')
recallxg=recall_score(Y_test,y_pred_xg,average='weighted')
fscorexg=f1_score(Y_test,y_pred_xg,average='weighted')
cmxg=confusion_matrix(Y_test,y_pred_xg)

################
#########################
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, Y_train)
 #Evaluate
Y_hat = [round(yhat) for yhat in model.predict(X_test)]
print(accuracy_score(Y_test, Y_hat)) # Test set accuracy
Y_hat = [round(yhat) for yhat in model.predict(X)]
print(accuracy_score(Y, Y_hat)) # Full set accuracy
udf = pd.read_csv('D:\\parkinsons-AI-master\\parkinsons_updrs.data')
udf.head()
features = udf.loc[:, (udf.columns != 'motor_UPDRS') & (udf.columns != 'total_UPDRS')].values[:, 1:]
labels = udf.loc[:, (udf.columns == 'motor_UPDRS') | (udf.columns == 'total_UPDRS')].values
X = scaler.fit_transform(features)
Y = scaler.fit_transform(labels)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=7)
from keras.models import Sequential
from keras.layers import Dense, LSTM
u_model = Sequential()
u_model.add(Dense(32, input_shape=(X.shape[1],)))
u_model.add(Dense(16, activation='tanh'))
u_model.add(Dense(8, activation='tanh'))
u_model.add(Dense(72, activation='tanh'))
u_model.add(Dense(Y.shape[1], activation='tanh'))
u_model.compile(optimizer='sgd', loss='mean_squared_error')
u_model.fit(X_train, Y_train, batch_size=1, epochs=5, validation_split=0.25, shuffle=True)
u_model.fit(X_train, Y_train, batch_size=1, epochs=15, validation_split=0.25, shuffle=True)
u_model.fit(X_train, Y_train, batch_size=1, epochs=75, validation_split=0.25, shuffle=True)
import matplotlib.pyplot as plt
Y_hat = u_model.predict(X_test)
error = np.abs((Y_hat - Y_test) / Y_test)
plt.show(plt.plot(error[:, 0][error[:, 0] > 3]))
plt.show(plt.plot(error[:, 1][error[:, 1] > 5]))
from xgboost import Booster
 
model._Booster.save_model('model.bin')
def load_xgb_model():
    _m = XGBClassifier()
    _b = Booster()
    _b.load_model('model.bin')
    _m._Booster = _b
    return _m

model = load_xgb_model()
from keras.models import load_model

u_model.save('u_model.hd5')
u_model = load_model('u_model.hd5')