
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import plotly.express as ex
import plotly.graph_objs as goo
import plotly.offline as pyo
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd


data = pd.read_csv(r'C:\Users\User\OneDrive - The American College of Greece\Documents\Εξόρυξη Δεδομένων\healthcare-dataset-stroke-data.csv')
print(data)



#A erwtima
figA = ex.pie(data, names='stroke')
figA.update_layout(title='<b>Proportion of Stroke Samples<b>')
#figA.show()

fig = make_subplots(
    rows=2, cols=2,subplot_titles=('','<b>Distribution Of Female Ages<b>','<b>Distribution Of Male Ages<b>','Residuals'),
    vertical_spacing=0.09,
    specs=[[{"type": "pie","rowspan": 2}       ,{"type": "histogram"}] ,
           [None                               ,{"type": "histogram"}]            ,
          ]
)

fig.add_trace(
    go.Pie(values=data.gender.value_counts().values,labels=['<b>Female<b>','<b>Male<b>','<b>Other<b>'],hole=0.3,pull=[0,0.08,0.3],marker_colors=['pink','lightblue','green'],textposition='inside'),
    row=1, col=1
)

fig.add_trace(
    go.Histogram(
        x=data.query('gender=="Female"').age,marker= dict(color='pink'),name='Female Ages'
    ),
    row=1, col=2
)
fig.add_trace(
    go.Histogram(
        x=data.query('gender=="Male"').age,marker= dict(color='lightblue'),name='Male Ages'
    ),
    row=2, col=2
)


fig.update_layout(
    height=800,
    showlegend=True,
    title_text="<b>Age-Sex Infrence<b>",
)

#fig.show()

plt.subplot(2,1,1)
plt.title('Stroke Sample Distribution Based On Bmi And Glucose Level')
sns.scatterplot(x=data['avg_glucose_level'], y=data['bmi'], hue=data['stroke'])
plt.subplot(2,1,2)
plt.title('Stroke Sample Distribution Based On Bmi And Age')
sns.scatterplot(x=data['age'], y=data['bmi'], hue=data['stroke'])
plt.tight_layout()
#plt.show()

stroke_population = data.query('stroke==1').copy()

fig = make_subplots(rows=2, cols=2, subplot_titles=('','<b>Distribution Of Female Ages<b>', '<b>Distribution Of Male Ages<b>', 'Residuals'), vertical_spacing=0.09, specs=[[{"type":"pie", "rowspan":2}, {"type":"histogram"}],[None,{"type":"histogram"}],])
fig.add_trace(go.Pie(values=stroke_population.gender.value_counts().values,labels=['<b>Female<b>','<b>Male<b>','<b>Other<b>'], hole=0.3, pull=[0,0.08,0.3], marker_colors=['pink','lightblue','green'],textposition='inside'), row=1, col=1)
fig.add_trace(goo.Histogram(x=stroke_population.query('gender=="Female"').age,marker=dict(color='pink'),name='Female Ages'), row=1, col=2)
fig.add_trace(goo.Histogram(x=stroke_population.query('gender=="Male"').age, marker=dict(color='lightblue'),name='Male Ages'), row=2, col=2)
fig.update_layout(height=800,showlegend=True,title_text="<b>Age-Sex Infrence Of Stroke Positive Samples<b>",)
#fig.show()

stroke_population = data.query('stroke ==1').copy()

fig= make_subplots(rows=2, cols=2, subplot_titles=('<b>Proportion Of Different Work Types<b>','<b>Proportion Of Married Individuals<b>','<b>Proportion Of Residence Type<b>','Residuals'),vertical_spacing=0.09,specs=[[{"type":"pie","rowspan":2}, {"type":"pie"}],[None,{"type":"pie"}],])
fig.add_trace(goo.Pie(values=stroke_population.work_type.value_counts().values,labels=['<b>Private<b>','<b>Self-employed<b>','<b>Govt_job<b>','<b>children<b>','<b>Never_worked<b>'],hole=0.3,pull=[0,0.08,0.03,0.2],marker_colors=['orange','green','blue','brown','purple'],textposition='inside'),row=1, col=1)
fig.add_trace(goo.Pie(values=stroke_population.ever_married.value_counts().values,labels=['<b>Yes<b>','<b>No<b>'],hole=0.3,pull=[0,0.08],marker_colors=['wheat','black'],textposition='inside'),
    row=1, col=2
)
fig.add_trace(
    go.Pie(values=stroke_population.Residence_type.value_counts().values,labels=['<b>Urban<b>','<b>Rural<b>'],hole=0.3,pull=[0,0.08],marker_colors=['pink','gray'],textposition='inside'),
    row=2, col=2
)
fig.update_layout(
    height=800,
    showlegend=True,
    title_text="<b>Different Categorical Attributes Of Stroke Samples<b>",
)

fig.show()

fig = ex.pie(stroke_population,names='smoking_status')
fig.update_layout(title='<b>Proportion Of Different Smoking Categories Among Stroke Population<b>')
#fig.show()

