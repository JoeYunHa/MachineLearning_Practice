import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

titanic = sns.load_dataset('titanic')

# print(titanic[['sex','embarked','who','deck','class','adult_male','embark_town','alive']].head())

# preprocess data
titanic = titanic.drop(columns=['embark_town','alive','class'])
titanic['who'] = titanic['who'].replace({
    'man':0,
    'woman':0,
    'child':1
}).astype('int64')

titanic['sex'] = titanic['sex'].replace({
    'male':0,
    'female':1
}).astype('int64')

# (C: Cherbourg, Q: Queenstown, S: Southampton)

embarked_encoded = pd.get_dummies(titanic['embarked'])
titanic = pd.concat([titanic, embarked_encoded], axis=1)
titanic = titanic.drop(columns=['embarked'])

deck_encoded = pd.get_dummies(titanic['deck'])
titanic = pd.concat([titanic, deck_encoded], axis=1)
titanic = titanic.drop(columns=['deck'])

titanic = titanic.dropna()

X = titanic.drop('survived', axis=1)
y = titanic['survived']

X.columns = X.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2)

# LinearSVC
s = LinearSVC(C=10)
s.fit(X_train, y_train)

res = s.predict(X_test)
print(res)

# get confusion matrix and visualized
conf = confusion_matrix(y_test, res)
print(conf)

sns.heatmap(conf, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
