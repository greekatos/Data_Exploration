import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style('darkgrid')
# import plotly.express as ex
# import plotly.graph_objs as goo
# import plotly.offline as pyo
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

data = pd.read_csv("healthcare-dataset-stroke-data.csv")
replace_gender = {'gender':{'Male':0, "Female":1, 'Other':2}}
replace_ever_married = {'ever_married':{'Yes':0, 'No':1}}
replace_work_type = {'work_type':{'Private':0, 'Self-employed':1, 'children':2, 'Govt_job':3, 'Never_worked':4}}
replace_Residence_type = {'Residence_type':{'Urban':0, 'Rural':1}}
replace_smoking_status = {'smoking_status':{'formerly smoked':0, 'never smoked':1, 'smokes':2, 'Unknown':3}}


#data[data.isnull().any(axis=1)]cleanup null values

data = data.replace(replace_gender)
data = data.replace(replace_ever_married)
data = data.replace(replace_work_type)
data = data.replace(replace_Residence_type)
data = data.replace(replace_smoking_status)

#data['bmi'].fillna((data['bmi'].mean()), inplace=True)

data.pop('id')
data.pop('smoking_status')
data.pop('bmi')
# data.pop('Residence_type')
# data.pop('work_type')
# data.pop('ever_married')
data_stroke = data.pop('stroke')


clf = RandomForestClassifier(random_state=100000)
clf.fit(data.head(3833), data_stroke.head(3833))
X = clf.predict(data.tail(1277))
np.savetxt('test.out', X, fmt="%d")
A = f1_score(data_stroke.tail(1277), X, average='weighted')
B = precision_score(data_stroke.tail(1277), X, average='weighted')
C = recall_score(data_stroke.tail(1277), X, average='weighted', zero_division=1)

print(A)
print(B)
print(C)
