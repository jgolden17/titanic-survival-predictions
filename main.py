"""
Predict the likelihood of surviving the Titanic sinkings
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

def get_training_data():
    """
    Get the training data set from Kaggle
    """
    train_url = 'http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv'
    training_data = pd.read_csv(train_url)
    training_data.fillna(training_data.mean(), inplace=True)
    training_data = training_data.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    label_encoder = LabelEncoder()
    label_encoder.fit(training_data['Sex'])
    training_data['Sex'] = label_encoder.transform(training_data['Sex'])

    return training_data

def main():
    """
    main
    """
    data = get_training_data()

    data_copy = data.copy()

    passengers_by_sex = np.array(data.drop(['Survived'], axis=1).astype(float))
    passengers_by_survival = np.array(data['Survived'])

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(passengers_by_sex)

    predictions = []

    correct = 0

    for index, passenger in enumerate(passengers_by_sex):
        predict = np.array(passenger.astype(float)).reshape(-1, len(passenger))
        prediction = kmeans.predict(predict)

        predictions.append(prediction)

        if prediction[0] == passengers_by_survival[index]:
            correct += 1

    print(correct/len(passengers_by_sex))

    data_copy['Prediction'] = predictions

    print(data_copy)

main()
