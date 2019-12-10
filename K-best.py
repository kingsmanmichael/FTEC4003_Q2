import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression


# 1. Read the training data and format it from csv to python dataframe.
train = pd.read_csv("credit-train.csv", header=None, sep=',')
train_headers = train.iloc[0]
train_df = pd.DataFrame(train.values[1:], columns=train_headers)

for col in train_df.columns:
    train_df[col] = pd.to_numeric(train_df[col])

# 2. Read the testing data into a DataFrame.
test = pd.read_csv("credit-test.csv", header=None, sep=',')
test_headers = test.iloc[0]
test_df = pd.DataFrame(test.values[1:], columns=test_headers)

for col in test_df.columns:
    test_df[col] = pd.to_numeric(test_df[col])

# 3. Train a random forest model.
X = train_df.iloc[:, :30]
y = train_df['Class']

# 4. Select 8 most related features.
k_best = SelectKBest(score_func=f_classif,k=8)
k_best.fit_transform(X,y)

mask = k_best.get_support()
feature_col = []

for bool, feature in zip(mask, train_df.columns):
    if bool:
        feature_col.append(feature)

# 5. Train the KNN model.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_df[feature_col],y)

# 6. Test the data using the trained KNN model.
test_pred = knn.predict(test_df[feature_col])
test_result = pd.DataFrame(test_pred.tolist(), columns=['Label'])
test_result.index.name = 'ID'
test_result.to_csv(r'submission2_4_kbest_knn.csv')
