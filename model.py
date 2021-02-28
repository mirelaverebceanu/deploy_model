from flask import Flask
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import json

app = Flask(__name__)


def predict():
    path = 'https://raw.githubusercontent.com/mirelaverebceanu/FIA/main/linear_regression/apartmentComplexData.csv'
    price_data = pd.read_csv(path)

    X = price_data[['column1','column2','complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr','column8']]
    Y = price_data[['medianCompexValue']]

    # Split X and y into X_
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=50)

    # create a Linear Regression model object.
    regression_model = LinearRegression()

    # pass through the X_train & y_train data set.
    regression_model.fit(X_train, y_train)
    train_model_score = regression_model.score(X_train, y_train)
    test_model_score = regression_model.score(X_test, y_test)
    value = y_test[0]
    predicted_value = regression_model.predict([X_test.iloc[0]])[0]

    return train_model_score, test_model_score, value, predicted_value, path


@app.route("/")
def index():
    return json.dumps({"Dataset": predict()[5]}, sort_keys=True)


@app.route("/modelScore")
def model_score():
    predict_result = predict()
    return json.dumps({"train_model_score": predict_result[0], "test_model_score": predict_result[1]}, sort_keys=True)


@app.route('/predict')
def value_prediction():
    predict_result = predict()
    return json.dumps({"predicted_value": predict_result[2], "value": predict_result[3]}, sort_keys=True)


if __name__ == '__main__':
    app.run()