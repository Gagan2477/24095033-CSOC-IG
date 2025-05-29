import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("KaggleV2-May-2016.csv")
df = df.drop(["PatientId", "AppointmentID"], axis=1)
df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['WaitTime'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df['WaitTime'] = df['WaitTime'].apply(lambda x: max(x, 0))  # Ensure no negatives
df = df.drop(['ScheduledDay', 'AppointmentDay'], axis=1)
df = pd.get_dummies(df, columns=['Gender', 'Neighbourhood'], drop_first=True)
X = df.drop("No-show", axis=1).values
y = df["No-show"].values.reshape(-1, 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
