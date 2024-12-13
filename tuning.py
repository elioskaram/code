import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('data.csv')
data = pd.get_dummies(data, columns=['Age', 'Sex'], drop_first=True)
X = data.drop(columns=['Murmur'])
y = data['Murmur']
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

##################################################### Tuning ########################################

print("Tuning(might take some time)")
n_estimators = [1, 10, 100, 1000, 10000]
max_leaf_nodes = [None, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for n in n_estimators:
    rf_classifier = RandomForestClassifier(n_estimators= n)
    rf_classifier.fit(X_train_scaled, y_train)
    predictions = rf_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print('For n_estimators = ', n)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

for m in max_leaf_nodes:
    rf_classifier = RandomForestClassifier(n_estimators= 1000, max_leaf_nodes=m)
    rf_classifier.fit(X_train_scaled, y_train)
    predictions = rf_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print('For max_leaf_nodes = ', m)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
