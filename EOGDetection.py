import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class EOGDetection:
    def __init__(self) -> None:
        pass


    def cleanup_data(self, weatherfilename, leaksfilename, sensorfilename):

        # clean input weather file
        weather = pd.read_csv(weatherfilename)
        cleaned_weather = weather.groupby("timestamp", as_index=False).mean()
        try:
            cleaned_weather.timestamp = cleaned_weather.timestamp.apply(lambda x: int(datetime.strptime(x, "%m/%d/%y %H:%M").timestamp()))
        except:
            cleaned_weather.timestamp = cleaned_weather.timestamp.apply(lambda x: int(datetime.strptime(x, "%m/%d/%Y %H:%M").timestamp()))

        cleaned_weather.to_csv("weather_cleaned.csv")

        # clean test leaks file
        leaks = pd.read_csv(leaksfilename)
        leaks = leaks.fillna(0)
        leaks["EmissionCategory"] = leaks["EmissionCategory"].apply(lambda x: "None" if x == 0 else x)
        remove = ["EventID", "UTCStart", "UTCEnd"]
        leaks.drop(remove, axis=1, inplace=True)
        leaks.to_csv("leaks_cleaned.csv")

        # clean input sensor file
        sensor = pd.read_csv(sensorfilename, index_col=0)
        sensor = sensor.rename(columns=lambda x: ",".join(x.split("_")[1:3]).strip() if x != "time" else x)
        sensor.to_csv("sensor_cleaned.csv")

        return {"cleaned_data": ["weather_cleaned.csv", "leaks_cleaned.csv", "sensor_cleaned.csv"]}



    def sensor_classification(self, sensorfilename, weatherfilename):

        df = pd.read_csv(sensorfilename, index_col=0)
        new = df.drop("time", axis=1)

        model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.1))

        model.fit(new)

        new["mean"] = new.mean(axis=1)

        new["score"] = model.decision_function(new.iloc[:, :-1])

        new["anomaly"] = model.predict(new.iloc[:, :-2])

        mean = new["mean"].mean(axis=0)

        new["timestamp"] = df["time"]

        weather = pd.read_csv(weatherfilename, index_col=0)

        weather = weather.set_index('timestamp')

        new = new.set_index("timestamp")

        new.loc[new['mean'] < mean, 'anomaly'] = 1

        joined = new.merge(weather, on="timestamp")

        return joined.T

    def predict_leaks(self, leaksfilename):


        leaks_train = pd.read_csv("data/leaks_trained.csv", index_col=0)
        leaks_test = pd.read_csv(leaksfilename, index_col=0)

        X_train = leaks_train.drop("LeakPointId", axis=1).values
        X_test = leaks_test.drop("LeakPointId", axis=1).values

        y_train = leaks_train["LeakPointId"].values
        y_test = leaks_test["LeakPointId"].values

        print(X_test)
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output = False), [3, 6])], remainder='passthrough')
        X_train = np.array(ct.fit_transform(X_train))
        X_test = np.array(ct.fit_transform(X_test))

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.fit_transform(y_test)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        confusion_matrix(y_test, y_pred)

        decoded_y_pred = le.inverse_transform(y_pred)

        leaks_test["MassRate"] = (leaks_test["Duration"] / 3600) * leaks_test["LeakRate"]

        test = np.hstack((leaks_test.values, np.reshape(decoded_y_pred, (-1, 1))))

        new = pd.DataFrame(test)

        new.drop([0, 1, 4, 5, 6, 7], axis=1, inplace=True)

        new.columns = ["Longitude", "Latitude", "Start", "End", "MassRate", "Leaks"] # break

        tmp = new.drop(["Longitude", "Latitude", "End"], axis=1)

        grouped_df = tmp.groupby('Start').agg({"Leaks": list, "MassRate": "mean"})

        grouped_df["Leak"] = grouped_df["Leaks"].apply(lambda x: [temp.split("-")[0] for temp in x])

        grouped_df = grouped_df.drop("Leaks", axis=1)

        grouped_df.reset_index(drop=False, inplace=True)

        message = ""
        for index, row in grouped_df.iterrows():
            if '0' in row['Leak']:
                message += f"{row['Start']}, None \n"
            elif len(row['Leak']) > 1:
                message += f"{row['Start']}, "
                for i in row['Leak']:
                    message += f"{i}|"
                message += "\n"

        return {"response": "Predicted leaks.", "data": grouped_df.T.to_dict(), "report": message, "accuracy": accuracy}


    def leak_status(self, leaksfilename):


        leaks_train = pd.read_csv("data/leaks_trained.csv", index_col=0)
        leaks_test = pd.read_csv(leaksfilename, index_col=0)

        X_train = leaks_train.drop("LeakPointId", axis=1).values
        X_test = leaks_test.drop("LeakPointId", axis=1).values

        y_train = leaks_train["LeakPointId"].values
        y_test = leaks_test["LeakPointId"].values

        print(X_test)
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output = False), [3, 6])], remainder='passthrough')
        X_train = np.array(ct.fit_transform(X_train))
        X_test = np.array(ct.fit_transform(X_test))

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.fit_transform(y_test)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        confusion_matrix(y_test, y_pred)

        decoded_y_pred = le.inverse_transform(y_pred)

        test = np.hstack((leaks_test.values, np.reshape(decoded_y_pred, (-1, 1))))

        new = pd.DataFrame(test)

        new.drop([0, 1, 4, 5, 6, 7], axis=1, inplace=True)

        new.columns = ["Longitude", "Latitude", "Start", "End", "Leaks"] # break

        return {"data": new.T.to_dict()}