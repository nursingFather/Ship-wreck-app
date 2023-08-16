import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="white", context="notebook", palette= "deep")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve,train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score

train = pd.read_csv("train.csv")
test= pd.read_csv("test.csv")

#print(train.head())

#print(train.info())

# create a new feature from  name
# convert to categorical values Title
def conv(dataset):
    dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
    dataset["Title"] = pd.Series(dataset_title)

    dataset["Title"] = dataset["Title"].replace(["Lady", "the Countess", "Countess", "Sir"], "Royal")
    dataset["Title"] = dataset["Title"].replace(["Capt", "Col", "Don", "Dr", "Major", "Rev", "Jonkheer", "Dona"],
                                                "Rare")
    dataset["Title"] = dataset["Title"].replace(["Mlle", "Ms"], "Miss")
    dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")

    g = sns.countplot(x="Title", data=dataset)
    g = plt.setp(g.get_xticklabels(), rotation=45)
conv(train)

print(train["Title"].unique())

index_NaN_age = list(train["Age"][train["Age"].isnull()].index)

for i in index_NaN_age:
    age_med = train["Age"].median()
    age_pred = train["Age"][((train["SibSp"] == train.iloc[i]["SibSp"]) &
                               (train["Parch"] == train.iloc[i]["Parch"]) &
                               (train["Pclass"] == train.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred):
        train["Age"].iloc[i]  = age_pred
    else:
        train["Age"].iloc[i] = age_med

# dropping the unnecessary  column
not_needed = ["Cabin","PassengerId", "Ticket", "Name"]

for col in not_needed:
    train.drop(col, axis=1, inplace=True)

df = train.dropna()
#for col in df.columns:
    #print(f"{col} : {df[col].unique()}")

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=11)

y_train = df_train.Survived.values
y_val = df_val.Survived.values

del df_train['Survived']
del df_val['Survived']

cat_col = df_train.select_dtypes(include=['object']).columns.tolist()
num_col = df_train.select_dtypes(include=['float64', "int64"]).columns.tolist()
print(cat_col)
print(num_col)

train_dict = df_train[cat_col + num_col].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)
#print(X_train)
train_dict = df_train[cat_col + num_col].to_dict(orient='records')

## Using Random Forest Classifier
rfc =  RandomForestClassifier()
rfc.fit(X_train, y_train)

# validation
val_dict = df_val[cat_col + num_col].to_dict(orient='records')
X_val = dv.transform(val_dict)
y_pred = rfc.predict_proba(X_val)[:, 1]

# Checking for Accuracy
#y_pred = rfc.predict_proba(X_val)[:, 1]
survive = y_pred >= 0.6
print((survive == y_val).mean())
print(f"Accuracy score:{accuracy_score(y_val, y_pred >= 0.6)}")
print(f"precision score:{precision_score(y_val, y_pred >= 0.6)}")
print(f"Recall score:{recall_score(y_val, y_pred >= 0.6)}")
print(f"F1 score:{f1_score(y_val, y_pred >= 0.6)}")

GBC = GradientBoostingClassifier()
LR = LogisticRegression(max_iter=1000)
Rfc =  RandomForestClassifier()

def train(model, df, y):
    features = df[cat_col + num_col].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    dv.fit(features)
    X = dv.transform(features)

    modell = model.fit(X, y)

    return dv, modell


def predict(df, dv, model):
    cat = df[cat_col + num_col].to_dict(orient='records')

    X = dv.transform(cat)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

import pickle

y_train = df_train_full.Survived.values
y_test = df_test.Survived.values

model_list = [GBC, LR, Rfc]
for mode in model_list:
    dv, modell = train(mode, df_train_full, y_train)
    y_pred = predict(df_test, dv, modell)

    auc = roc_auc_score(y_test, y_pred)
    print(f"The Auc score for {mode} is {' = %.3f' % auc}")
    with open(f'{mode}.bin', 'wb') as f_out:
        pickle.dump((dv, modell), f_out)
        f_out.close()


