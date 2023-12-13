import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# Adathalmaz betöltése
runs = pd.read_csv('runs.csv')
st.write(runs.head())

st.write(runs.isnull().any())
st.write(runs.shape)

# Adattisztítás
data = runs.drop(['time4', 'time5', 'time6', 'behind_sec4', 'behind_sec5', 'behind_sec6', 'position_sec4', 'position_sec5', 'position_sec6'], axis=1)
data = data.dropna()

st.write(data.head())
st.write(data.shape)

# Statisztikai jellemzők kiszámítása
st.subheader("Statisztikai információk:")
st.write(runs.describe())

# Train-test felosztás
y = data['result']
X = data.drop(['result', 'won', 'race_id', 'horse_no', 'horse_id', 'horse_country', 'horse_type', 'horse_gear'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

st.write('X_train', X_train.shape)
st.write('y_train', y_train.shape)
st.write('X_test', X_test.shape)
st.write('y_test', y_test.shape)

# RandomForestClassifier modell optimalizáció
st.subheader("LogisticRegression optimalizáció:")
y = data['result']
X = data.drop(['result', 'won', 'race_id', 'horse_no', 'horse_id', 'horse_country', 'horse_type', 'horse_gear'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

clf = RandomForestClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=20, n_estimators=800)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
st.write('Accuracy:', accuracy_score(y_test, y_pred))
st.write('Confusion matrix:', confusion_matrix(y_test, y_pred))
st.write(classification_report(y_test, y_pred))

lengths_behind = st.slider('Lengths Behind', min_value=0.0, max_value=999.0, value=10.0)
horse_age = st.slider('Horse Age', min_value=2, max_value=6, value=3)
horse_rating = st.slider('Horse Rating', min_value=0, max_value=200, value=50)
declared_weight = st.slider('Declared Weight', min_value=900, max_value=1200, value=1000)
actual_weight = st.slider('Actual Weight', min_value=110, max_value=200, value=125)
draw = st.slider('Draw', min_value=1, max_value=30, value=10)
position_sec1 = st.slider('Position Sec1', min_value=1, max_value=30, value=10)
position_sec2 = st.slider('Position Sec2', min_value=1, max_value=30, value=10)
position_sec3 = st.slider('Position Sec3', min_value=1, max_value=30, value=10)
behind_sec1 = st.slider('Behind Sec1', min_value=0.0, max_value=1000.0, value=50.0)
behind_sec2 = st.slider('Behind Sec2', min_value=0.0, max_value=1000.0, value=50.0)
behind_sec3 = st.slider('Behind Sec3', min_value=0.0, max_value=1000.0, value=50.0)
time1	= st.slider('Time1', min_value=0.0, max_value=1000.0, value=125.0)
time2 = st.slider('Time2', min_value=0.0, max_value=1000.0, value=125.0)
time3 = st.slider('Time3', min_value=0.0, max_value=1000.0, value=125.0)
finish_time	= st.slider('Finish time', min_value=30.0, max_value=200.0, value=125.0)
win_odds = st.slider('Win odds', min_value=0.0, max_value=100.0, value=50.0)
place_odds = st.slider('Place odds', min_value=0.0, max_value=1500.0, value=50.0)
trainer_id = st.slider('Trainer ID', min_value=1, max_value=200, value=125)
jockey_id = st.slider('Jockey ID', min_value=1, max_value=200, value=125)

prediction_features = [lengths_behind, horse_age, horse_rating, declared_weight, actual_weight, draw, position_sec1, position_sec2, position_sec3, behind_sec1, behind_sec2, behind_sec3, time1, time2, time3, finish_time, win_odds, place_odds, trainer_id, jockey_id]  # a további inputokat is hozzá kell adni
prediction = clf.predict([prediction_features])
st.subheader(f'The predicted result is: {[prediction[0]]}')
