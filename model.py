# Importing the libraries
import pandas as pd
import pickle

# Load the dataset
train = pd.read_csv('loan_train.csv')

# Handling the null values
train['LoanAmount'] = train['LoanAmount'].fillna(train['LoanAmount'].mean())
train['Credit_History'] = train['Credit_History'].fillna(train['Credit_History'].median())

train.dropna(inplace=True)

# Handling the categorical features
train['Loan_Status'].replace('Y', 1, inplace=True)
train['Loan_Status'].replace('N', 0, inplace=True)

train.Gender = train.Gender.map({'Male': 1, 'Female': 0})
train.Married = train.Married.map({'Yes': 1, 'No': 0})
train.Dependents = train.Dependents.map({'0': 0, '1': 1, '2': 2, '3+': 3})
train.Education = train.Education.map({'Graduate': 1, 'Not Graduate': 0})
train.Self_Employed = train.Self_Employed.map({'Yes': 1, 'No': 0})
train.Property_Area = train.Property_Area.map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Splitting train and test data
X = train.iloc[:, 1:12].values
y = train.iloc[:, 12].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting model with train data
regressor = LogisticRegression()
regressor.fit(X_train, y_train)

# Saving model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict(X_test))
