import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data.csv', sep=';')

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print("X_train before preprocessing:", X)
print("y_train before preprocessing:", y)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])
print("X after imputing missing values:", X)

# Encoding fitur kategori
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print("X after encoding categorical features:", X)

le = LabelEncoder()
y = le.fit_transform(y)

print("y after encoding:", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])

print("X_train after scaling:", X_train)
print("X_test after scaling:", X_test)