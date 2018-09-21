import csv
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X = []
Y = []


def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # skipping column names
        for row in csvFileReader:
            #print(', '.join(row))
            X.append(float(row[0]))
            Y.append(float(row[1]))
    return

def show_plot(X, Y):
    linear_mod = linear_model.LinearRegression()
    X = np.reshape(X, (len(X), 1))  # converting to matrix of n X 1
    Y = np.reshape(Y, (len(Y), 1))
    linear_mod.fit(X, Y)  # fitting the data points in the model
    plt.scatter(X, Y, color='yellow')  # plotting the initial datapoints
    plt.plot(X, linear_mod.predict(X), color='blue', linewidth=3)  # plotting the line made by linear regression
    plt.show()
    return

def predict_Y(X, Y, x):
    linear_mod = linear_model.LinearRegression()  # defining the linear regression model
    X = np.reshape(X, (len(X), 1))  # converting to matrix of n X 1
    Y = np.reshape(Y, (len(Y), 1))
    linear_mod.fit(X, Y)  # fitting the data points in the model
    predicted_Y = linear_mod.predict(x)
    return predicted_Y[0][0], linear_mod.coef_[0][0], linear_mod.intercept_[0]

get_data('slr04.csv')  # calling get_data method by passing the csv file to it
print("\n", X)

print("\n")

print("\n",Y)

show_plot(X, Y)

print("predicted values of Y are:")
(predicted_Y, coefficient, constant) = predict_Y(X, Y, 1)
print("The predicted value of Y is: $", str(predicted_Y))
print("The regression coefficient is ", str(coefficient), ", and the constant is ", str(constant))
print("the relationship equation between X and Y is: Y = ", str(coefficient), "* X + ", str(constant))


# linear_mod.coef_ # to see the coefficients
#
# linear_mod.scores(X,y) # To see the scores
#
#
# linear_mod.intercept_ # To display the intercept
