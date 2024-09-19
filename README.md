# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1. Start.

step 2. Import Necessary Libraries and Load Data.

step 3. Split Dataset into Training and Testing Sets.

step 4. Train the Model Using Stochastic Gradient Descent (SGD).

step 5. Make Predictions and Evaluate Accuracy.

step 6. Generate Confusion Matrix.

step 7. Stop.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Pranavesh Saikumar
RegisterNumber: 212223040149
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test,y_pred)
print(f"Confusion Matrix:")
print(cm)
```

## Output:

![image](https://github.com/user-attachments/assets/9f14766a-2bbc-4f14-b6d4-e8bfabe18178)

![image](https://github.com/user-attachments/assets/e3cbe50b-c585-4412-a5a0-7a4ecc755b26)

![image](https://github.com/user-attachments/assets/4f0cdd92-e033-410f-80f9-aef4edd837cd)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
