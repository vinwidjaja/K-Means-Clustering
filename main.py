import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

baca = pd.read_excel('Titanic.xls')
print(baca.head())

baca.fillna(0, inplace = True)

def handle_non_numerical_data(baca):
    columns = baca.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if baca[column].dtype != np.int64 and baca[column].dtype != np.float64:
            column_contents = baca[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            baca[column] = list(map(convert_to_int, baca[column]))
    return baca

baca = handle_non_numerical_data(baca)
print(baca.head())

baca.drop(['ticket'],1 , inplace=True)

X = np.array(baca.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(baca['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
        
print(float(correct)/float(len(X)))
    
