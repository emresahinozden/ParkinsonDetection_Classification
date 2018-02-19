import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#Under "Archived users" folder there are 227 users and their respective information
#I extract and append them in a dataframe as a start

users = []

for root, dirs, names in os.walk("C:\\Users\\MONSTER\\Downloads\\ML\\parkinson\\Archived users", topdown=False):
   for name in names:
      users.append(name[5:15])

UserFrame = pd.DataFrame(users, columns = ["Users"])

val = []
values = []

for i in range(len(users)):
    path = "C:\\Users\\MONSTER\\Downloads\\ML\\parkinson\\Archived users\\" + names[i]
    text_file = open(path)
    lines = text_file.read().split("\n")
    columns = []
    for j in range(len(lines)-1):
        x = lines[j].split(": ",1)
        v = x[1]
        c = x[0]
        val.append(v)
        columns.append(c)
    values.append(val)
    val = []
    
ValueFrame = pd.DataFrame(values, columns=columns)

Mat1 = pd.concat([UserFrame, ValueFrame], axis=1)

#Under "Tappy Data" folder there are 621 text files regarding the keyboard type movements of the users
#I extract and append them in a dataframe similarly, 
#but slightly different from because their structure is also different from the previous user files

usersTappy = []

for root, dirs, names in os.walk("C:\\Users\\MONSTER\\Downloads\\ML\\parkinson\\Tappy Data", topdown=False):
   for name in names:
      usersTappy.append(name[0:10])

UserFrameTappy = pd.DataFrame(usersTappy, columns = ["Users"])

val = []

for i in range(len(usersTappy)):
    path = "C:\\Users\\MONSTER\\Downloads\\ML\\parkinson\\Tappy Data\\" + names[i]
    text_file = open(path)
    lines = text_file.read().split("\n")
    for j in range(len(lines)-1):
        x = lines[j]
        x = x.split("\t")
        val.append(x)

ValueFrameTappy = pd.DataFrame(val, columns=["Users","Date","Time","Key Location","5","Movement","7","8","9"])

#I join the two dataframes in one big dataframe and then after I start to clean data
Matrix = pd.merge(ValueFrameTappy, Mat1, how='inner', left_on='Users', right_on='Users')

#I remove the users that has less than 2000 keystrokes (also as done in the study itself)
#Ref: The complete dataset comprised 217 participants, 
#however only some of those were included the subsequent analysis, comprising:
#Those with at least 2000 keystrokes
counts = Matrix['Users'].value_counts()
Matrix = Matrix[Matrix['Users'].isin(counts[counts > 2000].index)]
Matrix = Matrix[Matrix['Users'].isin(counts[counts < 100000].index)]

#Ref: Of the ones with PD, just the ones with ‘Mild’ severity 
#(since the study was into the detection of PD at its early stage, not later stages)
Matrix = Matrix[Matrix.Impact != "Severe"]
Matrix = Matrix[Matrix.Impact != "Medium"]

#Ref: Those not taking levodopa
Matrix = Matrix[Matrix.Levadopa == "False"]

#I drop the irrelevant columns such as Users or Levadopa and Impact since those columns are consisted of all same value after filtering
Matrix = Matrix.drop(["Users", "Date", "Time", "8", "9","UPDRS","Impact","Levadopa","Tremors", "Other","Sided","DA","MAOB","DiagnosisYear" ], axis=1)
Matrix = Matrix.rename(columns={"5": "Hold", "7": "Latency"})
#The space is a mutual key (can not be defined as L or R), so I only include the directional (LL, LR, RR, RL) movements
Matrix = Matrix[(Matrix.Movement == "LL") | (Matrix.Movement == "LR") | (Matrix.Movement == "RR") | (Matrix.Movement == "RL") ]
Matrix = Matrix[(Matrix.BirthYear != "")]

#Now I deal with the categorical data: Key Location, Movement, Gender, Sided, Parkinsons
#I start by numeralize them, and when all the values are numeralized I will scale them in order to create
#a balanced numeric dataframe. Otherwise a column's bigger magnitudes can be over-affecting the calculations in algorithms.
Matrix.Gender.replace(['Male', 'Female'], [1, 0], inplace=True)
Matrix.Parkinsons.replace(['True', 'False'], [1, 0], inplace=True)
Matrix = Matrix.rename(columns={"Key Location": "RightKey"})
Matrix.RightKey.replace(['R', 'L'], [1, 0], inplace=True)
Matrix.Movement.replace(['LL', 'LR', 'RR', 'RL'], [0, 1, 2, 3], inplace=True)
Matrix.BirthYear = Matrix.BirthYear.astype(int)
Matrix.BirthYear = 2017 - Matrix.BirthYear
Matrix = Matrix.rename(columns={"BirthYear": "Age"})
Matrix = Matrix[pd.to_numeric(Matrix['Latency'], errors='coerce').notnull()]
Matrix.Latency = Matrix.Latency.astype(float)
Matrix.Hold = Matrix.Hold.astype(float)

#Now, since the Movement fields are named 1,2,3,4 they can be mistaken for greater or lesser even dough there is no hierarchy among them
#To cope with it, I use dummy encoding. Finally replace it with the previous column
Mov = Matrix.Movement
DummyMov = pd.get_dummies(Mov)
DummyMov = DummyMov.rename(columns={0: "LL", 1: "LR", 2: "RR", 3: "RL"})
Matrix = Matrix.drop(["Movement"], axis=1)
Matrix = pd.concat([Matrix, DummyMov], axis=1)
Matrix = Matrix.reindex_axis(["RightKey","LL","LR","RR","RL","Latency","Hold","Age","Gender","Parkinsons"], axis=1)

#I will not use these variables anymore, I delete them to save memory and for a clean working environment
del DummyMov, Mov, ValueFrameTappy, val, i, j, UserFrame, UserFrameTappy, ValueFrame, Mat1, c, columns, counts, dirs, lines, name, names, path, root, users, usersTappy, v, x, values

#I split the dataset into response variable and values
X = Matrix.iloc[:, :-1].values
y = Matrix.iloc[:, -1].values

#I split the X and y into training and test sets, I use scikit learn. I also define a subset of data for visualization.
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

Matrix_graph = Matrix.sample(n=100)
X_graph = Matrix_graph.iloc[:, 5:7].values
y_graph = Matrix_graph.iloc[:, -1].values

del Matrix, X, y, Matrix_graph

#Now, finally before fitting the classification model, I do the scaling. I again use scikit learn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_graph = scaler.fit_transform(X_graph)

#---------------------------------------------------------------------------------------------------------------------------------------

#Now we are good to go, but there is one final consideration:
#X_train is 600.000 rows and X_test is 150.000 rows. It will be very demanding to make computations on these large scales(Especially SVM)
#So I reduce the size of datasets from 750.000 to 30.000 randomly by applying (0.2) two times:
X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size = 0.2)
X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size = 0.2)

#First I use Logistic Regression for a model fitting
from sklearn.linear_model import LogisticRegression
classifierLogReg = LogisticRegression(random_state = 0)
classifierLogReg.fit(X_train, y_train)

#The algorithm has learnt on training set, now I will predict the outcomes on test set
y_pred_LogReg0 = classifierLogReg.predict(X_train)
y_pred_LogReg = classifierLogReg.predict(X_test)

#I compare the predicted results with the actual results, I use a confussion matrix to easily see the types and errors
from sklearn.metrics import confusion_matrix
CM_LogReg0 = confusion_matrix(y_train, y_pred_LogReg0)
PercLogReg0 = (CM_LogReg0[0,0]+CM_LogReg0[1,1])/CM_LogReg0.sum()

CM_LogReg = confusion_matrix(y_test, y_pred_LogReg)
PercLogReg = (CM_LogReg[0,0]+CM_LogReg[1,1])/CM_LogReg.sum()


#Next I use KNN to fit algorithm
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)

y_pred_KNN0 = classifierKNN.predict(X_train)
y_pred_KNN = classifierKNN.predict(X_test)

CM_KNN0 = confusion_matrix(y_train, y_pred_KNN0)
PercKNN0 = (CM_KNN0[0,0]+CM_KNN0[1,1])/CM_KNN0.sum()

CM_KNN = confusion_matrix(y_test, y_pred_KNN)
PercKNN = (CM_KNN[0,0]+CM_KNN[1,1])/CM_KNN.sum()


#Next I use SVM Kernal to fit algorithm
from sklearn.svm import SVC
classifierSVC = SVC(kernel = 'rbf', random_state = 0)
classifierSVC.fit(X_train, y_train)

y_pred_SVMK0 = classifierSVC.predict(X_train)
y_pred_SVMK = classifierSVC.predict(X_test)

CM_SVMK0 = confusion_matrix(y_train, y_pred_SVMK0)
PercSVMK0 = (CM_SVMK0[0,0]+CM_SVMK0[1,1])/CM_SVMK0.sum()

CM_SVMK = confusion_matrix(y_test, y_pred_SVMK)
PercSVMK = (CM_SVMK[0,0]+CM_SVMK[1,1])/CM_SVMK.sum()


#Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
classifierBAYES = GaussianNB()
classifierBAYES.fit(X_train, y_train)

y_pred_BAYES0 = classifierBAYES.predict(X_train)
y_pred_BAYES = classifierBAYES.predict(X_test)

CM_BAYES0 = confusion_matrix(y_train, y_pred_BAYES0)
PercBAYES0 = (CM_BAYES0[0,0]+CM_BAYES0[1,1])/CM_BAYES0.sum()

CM_BAYES = confusion_matrix(y_test, y_pred_BAYES)
PercBAYES = (CM_BAYES[0,0]+CM_BAYES[1,1])/CM_BAYES.sum()


#Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
classifierTREE = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierTREE.fit(X_train, y_train)

y_pred_TREE = classifierTREE.predict(X_test)
y_pred_TREE0 = classifierTREE.predict(X_train)

CM_TREE0 = confusion_matrix(y_train, y_pred_TREE0)
PercTREE0 = (CM_TREE0[0,0]+CM_TREE0[1,1])/CM_TREE0.sum()

CM_TREE = confusion_matrix(y_test, y_pred_TREE)
PercTREE = (CM_TREE[0,0]+CM_TREE[1,1])/CM_TREE.sum()


#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifierFOR = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierFOR.fit(X_train, y_train)

y_pred_FOR0 = classifierFOR.predict(X_train)
y_pred_FOR = classifierFOR.predict(X_test)

CM_FOR0 = confusion_matrix(y_train, y_pred_FOR0)
PercFOR0 = (CM_FOR0[0,0]+CM_FOR0[1,1])/CM_FOR0.sum()

CM_FOR = confusion_matrix(y_test, y_pred_FOR)
PercFOR = (CM_FOR[0,0]+CM_FOR[1,1])/CM_FOR.sum()


#Ensemble voting

from sklearn.ensemble import VotingClassifier
Models = []
Models.append(('KNN', classifierKNN))
#Models.append(('SVC', classifierSVC))
Models.append(('Tree', classifierTREE))

ensemble = VotingClassifier(Models)
ensemble = ensemble.fit(X_train, y_train)

y_pred_ENS0 = ensemble.predict(X_train)
y_pred_ENS = ensemble.predict(X_test)

CM_ENS0 = confusion_matrix(y_train, y_pred_ENS0)
PercENS0 = (CM_ENS0[0,0]+CM_ENS0[1,1])/CM_ENS0.sum()

CM_ENS = confusion_matrix(y_test, y_pred_ENS)
PercENS = (CM_ENS[0,0]+CM_ENS[1,1])/CM_ENS.sum()

#-------------------------------------------------------------------------------------------------
#Now I visualize the logistic Regression, a better fiting model: KNN, and best fitting model: Random Forest
#I get these virtualization codes from internet and I dont have a deep understanding on plt.contourf
#The two variable I will use in virtualizing will be Hold and Latency

#Logistic Regression Visualization

from matplotlib.colors import ListedColormap

classifierLogReg_graph = LogisticRegression(random_state = 0)
classifierLogReg_graph.fit(X_graph, y_graph)

X_set, y_set = X_graph, y_graph
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.figure(1)
plt.contourf(X1, X2, classifierLogReg_graph.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression')
plt.xlabel('Latency')
plt.ylabel('Hold')
plt.legend()
plt.show()


#KNN Visualization
classifierKNN_graph = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKNN_graph.fit(X_graph, y_graph)

X_set, y_set = X_graph, y_graph
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.figure(2)
plt.contourf(X1, X2, classifierKNN_graph.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN')
plt.xlabel('Latency')
plt.ylabel('Hold')
plt.legend()
plt.show()


#SVM Visualization
classifierSVC_graph = SVC(kernel = 'rbf', random_state = 0)
classifierSVC_graph.fit(X_graph, y_graph)

X_set, y_set = X_graph, y_graph
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.figure(3)
plt.contourf(X1, X2, classifierSVC_graph.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM')
plt.xlabel('Latency')
plt.ylabel('Hold')
plt.legend()
plt.show()


#Random Forest Visualization
classifierFOR_graph = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierFOR_graph.fit(X_graph, y_graph)

X_set, y_set = X_graph, y_graph
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.figure(4)
plt.contourf(X1, X2, classifierFOR_graph.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest')
plt.xlabel('Latency')
plt.ylabel('Hold')
plt.legend()
plt.show()
close(fig)
