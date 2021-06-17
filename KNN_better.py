import numpy as np
import pandas as pd
from knn import knn_impute
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style('darkgrid')
# import plotly.express as ex
# import plotly.graph_objs as goo
# import plotly.offline as pyo
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go mybro is so cool and i love him <3

data = pd.read_csv("healthcare-dataset-stroke-data.csv")
replace_gender = {'gender':{'Male':0, "Female":1, 'Other':2}}
replace_ever_married = {'ever_married':{'Yes':0, 'No':1}}
replace_work_type = {'work_type':{'Private':0, 'Self-employed':1, 'children':2, 'Govt_job':3, 'Never_worked':4}}
replace_Residence_type = {'Residence_type':{'Urban':0, 'Rural':1}}
replace_smoking_status = {'smoking_status':{'formerly smoked':0, 'never smoked':1, 'smokes':2, 'Unknown':3}}
n= 75
q= 25



data = data.replace(replace_gender)
data = data.replace(replace_ever_married)
data = data.replace(replace_work_type)
data = data.replace(replace_Residence_type)
data = data.replace(replace_smoking_status)



data['bmi']=knn_impute(target=data['bmi'], attributes=data.drop(['bmi'],1), k_neighbors=5)
i = data[data.id == 2019].index
data = data.drop(i)

# data['bmi'].interpolate(method='linear', inplace=True) auto einai gia to Linear
# data = data.sample(frac=1).reset_index(drop=True)
np.savetxt('test.out', data, fmt="%10.5f")
# data.pop('id')
# data.pop('smoking_status')
# data.pop('bmi')
# data.pop('Residence_type')
# data.pop('work_type')
# data.pop('ever_married')
data_stroke = data.pop('stroke')


clf = RandomForestClassifier(random_state=True, n_estimators=100,class_weight='balanced')
clf.fit(data.head(int(len(data)*(n/100))), data_stroke.head(int(len(data)*(n/100))))
X = clf.predict(data.tail(int(len(data)*(q/100))))

A = f1_score(data_stroke.tail(int(len(data)*(q/100))), X, average='micro')
B = precision_score(data_stroke.tail(int(len(data)*(q/100))), X, average='micro')
C = recall_score(data_stroke.tail(int(len(data)*(q/100))), X, average='micro', zero_division=1)

errors = abs(X - data_stroke.tail(int(len(data)*(q/100))))
# print(round(np.mean(errors), 2))


print(A)
print(B)
print(C)


