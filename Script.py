import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 1. Read the training data into a DataFrame.
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

X = train_df.iloc[:, :30]
y = train_df['Class']

# 3. Train a random forest model.
rft = RandomForestClassifier(n_estimators=100)
rft.fit(X, y)

# 4. Select the 15 most important features.
feature_imp = pd.Series(rft.feature_importances_,index=train_df.iloc[:, :30].columns).sort_values(ascending=False)
feature_col = feature_imp[:15].index.tolist()
new_X = train_df[feature_col]

# 5. Plot the 15 most important bar chart.
fig = plt.figure(figsize=(20, 6))
plt.bar(feature_imp.iloc[:15].index, feature_imp.iloc[:15].values, align='center')
plt.title("The 15 Important Features")
plt.xlabel("Features")
plt.rcParams.update({'font.size': 12})

# 6. Train a new random forest model by just inputting the feature attributes.
rft2 = RandomForestClassifier(n_estimators=30)
rft2.fit(new_X, y)

# 7. Train a KNN model with selected features.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(new_X,y)

# 8. Predict the outcome using the first random forest model.
test_pred = rft.predict(test_df)
test_result = pd.DataFrame(test_pred.tolist(), columns=['Label'])
test_result.index.name = 'ID'
test_result.to_csv(r'submission2_1_rft.csv')

# 9. Predict the outcome using the second random forest model.
test_pred2 = rft2.predict(test_df[feature_col])
test_result2 = pd.DataFrame(test_pred2.tolist(), columns=['Label'])
test_result2.index.name = 'ID'
test_result2.to_csv(r'submission2_2_rft_rft.csv')

# 10. Predict the outcome using the KNN model.
test_pred3 = knn.predict(test_df[feature_col])
test_result3 = pd.DataFrame(test_pred3.tolist(), columns=['Label'])
test_result3.index.name = 'ID'
test_result3.to_csv(r'submission2_3_rft_knn.csv')
