import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from null_checker import missing_data
from sklearn import tree


"""
ここを参考にサンプルプログラムを作ってみる（https://www.codexa.net/kaggle-titanic-beginner/）
"""
# load titanic data
train = pd.read_csv('inputs/train.csv')
test = pd.read_csv('inputs/test.csv')

# 欠損データの確認
missing_data_train = missing_data(train)
missing_data_test = missing_data(test)

# --------------以下，データの前処理--------------
# 欠損データを代理データに入れ替える(中央値を最頻値を用いた)
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")
missing_table_train = missing_data(train)

# 文字列を数字に変換
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S" ] = 0
train["Embarked"][train["Embarked"] == "C" ] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# テストデータも同様に
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.median()

# 「train」の目的変数と説明変数の値を取得
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# 決定木の作成
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# 「test」の説明変数の値を取得
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# 「test」の説明変数を使って「my_tree_one」のモデルで予測
my_prediction = my_tree_one.predict(test_features)

# PassengerIdを取得
PassengerId = np.array(test["PassengerId"]).astype(int)

# my_prediction(予測データ）とPassengerIdをdfへ落とし込む
my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])

# my_tree_one.csvとして書き出し
my_solution.to_csv("outputs/my_tree_one.csv", index_label=["PassengerId"])