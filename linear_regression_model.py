# IMPORTING LIBRARIES AND DEPENDENCIES
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

# LOADING IN TRAINING DATA SET
data = pd.read_csv('student-mat.csv', sep=';')

# GETTING RELEVANT COLUMNS FROM DATA SET
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# SEPARATING PREDICTOR AND TARGET VALUES
target = 'G3'
X = data.drop([target], 1).values
Y = data[target].values

# SPLITTING X AND Y INTO TRAINING AND TEST SAMPLES
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

# INITIALIZING LINEAR MODEL
model = LinearRegression()

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    model = LinearRegression()

    model.fit(x_train, y_train)

    # GET MODEL ACCURACY
    acc = model.score(x_test, y_test) * 100

    # IF CURRENT MODEL HAS A BETTER SCORE SAVE IN A PICKLE FILE
    if acc > best:
        best = acc
        with open("model.pickle", "wb") as f:
            pickle.dump(model, f)

# LOAD MODEL
# pickle_in = open("model.pickle", "rb")
# linear = pickle.load(pickle_in)
#
#
# # CALCULATE SOME METRIC DATA
# metrics = pd.Series({
#     "Accuracy" : linear.score(x_test, y_test) * 100,
#     'Co-efficient' : linear.coef_,
#     'Intercept' : linear.intercept_
#
# }, )
#
# predicted = linear.predict(x_test)
#
# # DRAWING AND PLOTTING THE MODEL WITH ITS PREDICTIONS
# df = pd.DataFrame({'Actual' : y_test.flatten(), 'Predicted' : predicted.flatten()})

# IN ORDER TO PLOT DIFFERENCE
# df1 = df.head(20)
# df1.plot(kind='bar',figsize=(16,10))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.xlabel('Accuracy : ' + str(metrics['Accuracy']))
# plt.show()
