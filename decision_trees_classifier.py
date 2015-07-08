# -*- coding: utf-8 -*- #
"""
Created on Mon Jul 06 22:16:03 2015

@author: Allen Thomas Varghese
"""
import pandas as pd
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot


# Convert strings to integer values with a lookup dictionary
outlook_values = {'sunny': 1, 'overcast': 2, 'rain': 3, 'storm': 4, 'thunder': 5}
temperature_values = {'hot': 1, 'mild': 2, 'cool': 3, 'freezing': 4}
humidity_values = {'high': 1, 'normal': 2, 'low': 3, 'dry': 4}
windy_values = {False: 1, True: 2}


def create_decision_tree():
    print "\nCreating Decision Tree..."
    # Read CSV values into a pandas DataFrame.
    # Use skipinitial space to remove the leading space
    df = pd.read_csv('weather.csv', skipinitialspace=True)

    df['outlook'] = map(lambda x: outlook_values[x], df['outlook'])
    df['temperature'] = map(lambda x: temperature_values[x], df['temperature'])
    df['humidity'] = map(lambda x: humidity_values[x], df['humidity'])
    df['windy'] = map(lambda x: windy_values[x], df['windy'])

    print "Input Data from CSV:\n", df

    # Create the feature set
    X = map(list, df[['outlook', 'temperature', 'humidity', 'windy']].values)

    # Create the output values
    Y = df['class'].tolist()

    # Default criteria for quality of split is 'gini'
    # clf = tree.DecisionTreeClassifier()

    # Setting the criteria for quality of split to 'entropy'
    clf = tree.DecisionTreeClassifier(criterion='entropy')

    clf = clf.fit(X, Y)

    print "Decision Tree created!"

    make_predictions(clf)

    return clf


def make_predictions(clf):
    print "\nRunning predictions..."
    # Verification: Check if a prediction matches with an existing value
    # Expected Output: 'N'
    print "Input Values: ['storm', 'cool', 'high', True]"
    print "Prediction: ", clf.predict([
        [
            outlook_values['storm'],
            temperature_values['cool'],
            humidity_values['high'],
            windy_values[True]
        ]
    ])

    # Predict a value that is not available in the input data
    # Expected Output: 'P'
    print "Input Values: ['overcast', 'mild', 'normal', True]"
    print "Prediction: ", clf.predict([
        [
            outlook_values['overcast'],
            temperature_values['mild'],
            humidity_values['normal'],
            windy_values[True]
        ]
    ])
    print "Predictions generated!"


def generate_plot(clf):
    print "\nGenerating plot..."
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("weather_forecast.pdf")
    print "Plot generated!"


if __name__ == '__main__':
    print "Staring execution..."
    classifier = create_decision_tree()
    generate_plot(classifier)
    print "\nExecution Completed!"
