import streamlit as st

st.title("Projekt PAD. S18889")

#---- czyszczenie ----#
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly as plo
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.feature_selection import SequentialFeatureSelector
import time

#Wczytanie danych
stones = pd.read_csv('messy_data.csv', skipinitialspace=True)

#Usunięcie duplikatów
stones = stones.drop_duplicates()
#Zamieniam wartoście NAN na średnią
stones['carat'].fillna(stones['carat'].mean(), inplace = True)
stones['table'].fillna(stones['table'].mean(), inplace = True)

#Usuwam kolumny z zbyt dużą ilością NAN
stones.drop(columns=['y dimension','depth'], inplace=True)
#'z dimension' nie bardzo dużo NAN ale jest mocno skorelowane z carat więc uzanłem że warto usunąć.
stones.drop(columns='z dimension', inplace=True)

#Usuwam wiersze z NAN w przypadku kolumn gdzie NAN jest nieczęste
stones.dropna(subset=['x dimension','price'], axis='index', inplace=True)

#Usuwam wartości odstające
stones[stones['carat']>5] = stones['carat'].mean()
stones = stones[stones.price<50000]
stones = stones[(stones.price!=3400) | (stones.carat!=1.6)]

#Wielkie litery na małe.
stones['color'] = stones['color'].str.lower()
stones['cut'] = stones['cut'].str.lower()
stones['clarity'] = stones['clarity'].str.lower()

#przed zamianą z kategorycznych na numeryczne robię kopię. W celu wizualizacji
stones_categorical = stones.copy()


stones_coppy = stones

#Zamieniam zmienną kategoryczną color na medianę wartości
median = stones_coppy.groupby(stones['color'])['price'].median()
stones['color'] = stones_coppy.groupby(stones['color'])['color'].transform(lambda category: median[category.name] )

#Zamieniam zmienną kategoryczną cut na medianę wartości
median = stones_coppy.groupby(stones['cut'])['price'].median()
stones['cut'] = stones_coppy.groupby(stones['cut'])['cut'].transform(lambda category: median[category.name] )

#Zamieniam zmienną kategoryczną clarity na medianę wartości
median = stones_coppy.groupby(stones['clarity'])['price'].median()
stones['clarity'] = stones_coppy.groupby(stones['clarity'])['clarity'].transform(lambda category: median[category.name] )

#---- Wizualizacja ----#
st.title("Wizualizacja danych")
#Rozkład zmiennych
st.subheader("histogram ceny z wyszczegulnionymi klasami Atrybutów kategorycznych")
Selected = st.radio("Wybierz atrybut do rozkładu",('color','cut','clarity') )

fig = px.histogram(stones_categorical, x='price', color=Selected)
st.plotly_chart(fig)

#Średnie wartości klas.
st.subheader("Mediana wartości dla kategori")
Selected = st.selectbox("Wybierz atrybut do mediany",('color','cut','clarity') )

stones_coppy = stones_categorical
medians = stones_coppy.groupby(stones_categorical[Selected])['price'].median().sort_values()
fig = px.bar(medians,x='price')
st.plotly_chart(fig)

#Histogramy
st.subheader("Histogram")
Selected = st.selectbox("Wybierz atrybut do histogramu",('carat','clarity','color','cut','x dimension','table','price') )

stones_coppy = stones_categorical
fig = px.histogram(stones_categorical,x=Selected)
st.plotly_chart(fig)

#Wykresy pudełkowe
st.subheader("Wykresy pudełkowe")
Selected = st.multiselect("Wybierz atrybut do wykresu pudełkowego",('table','price','x dimension','carat') )

fig = sp.make_subplots(rows=1, cols=len(Selected), shared_xaxes=True)
for i, column in enumerate(Selected):
    fig.add_trace(go.Box(y=stones[column], name=column), row=1, col=1+i)

fig.update_layout(title='')
st.plotly_chart(fig)

#Wykresy scater
st.subheader("Wykresy zaleczności")
col1, col2 = st.columns(2)
with col1:
    Selected_x = st.selectbox("Wybierz oś x wykresu",('carat','clarity','color','cut','x dimension','table','price') )
with col2:
    Selected_y = st.selectbox("Wybierz oś y wykresu",('carat','clarity','color','cut','x dimension','table','price') )

fig = px.scatter(stones_categorical,x=Selected_x,y=Selected_y,color='price')
st.plotly_chart(fig)

#macierz korelacji
st.subheader("Macierz korelacji")
Selected = st.multiselect("Wybierz atrybuty do macierzy korelacji",('carat','clarity','color','cut','x dimension','table','price') )
correlation_matrix = stones[Selected].corr()

fig = px.imshow(
    correlation_matrix,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    title='Correlation matrix'
)

st.plotly_chart(fig)


#---- modelowanie ----#
st.title("Wizualizacja Modelu")
X_train, X_test, Y_train, Y_test = train_test_split(stones.drop(columns=['price']),stones['price'],test_size=0.33)
LR = LinearRegression()
scores = []
for i in range(1,6):
    SFS = SequentialFeatureSelector(LR, n_features_to_select=i, scoring='r2')
    SFS = SFS.fit(X_train,Y_train)
    selected_feature_indices = SFS.get_support()
    #X_train.columns[selected_feature_indices]

    score = cross_val_score(estimator=LR,X = X_train.iloc[:, SFS.get_support()],y=Y_train,cv=10).mean()
    scores.append(score)

no_atributes = st.slider('wybierz Ilość atrybutów',min_value=1,max_value=5)
fig = px.scatter(x=range(1,6) , y=scores,title='Model',labels={'x':'Number of selected features','y':'R2 score'})
fig.add_vline(x=no_atributes, line_dash="dash", line_color="green")
fig.update_layout(showlegend=False)
st.subheader("Model zalerzność R2 od ilości wybranych atrybutów")
st.plotly_chart(fig)

with st.spinner("Trenowanie modelu"):
    #Dodałem żeby było widać że działa. :P 
    time.sleep(1.5)
    SFS = SequentialFeatureSelector(LR, n_features_to_select=no_atributes, scoring='r2')
    SFS = SFS.fit(X_train,Y_train)
    selected_feature_indices2 = SFS.get_support()
    X_train.iloc[:, selected_feature_indices2]

    selected = X_train.iloc[:, selected_feature_indices2]
    LR = LinearRegression()
    model = LR.fit(selected,Y_train)
    predicted = model.predict(X_test.iloc[:, selected_feature_indices2])
    st.subheader("R2 score")
    st.write(r2_score(Y_test,predicted))

    #przygotowanie wników do wizualizacji
    predicted = pd.DataFrame(predicted,columns=['predicted'])
    stones_predicted = pd.concat([X_test,Y_test],axis='columns')
    stones_predicted = pd.concat([stones_predicted.reset_index(drop=True),predicted.reset_index(drop=True)],axis='columns')

    st.subheader("Przewidziane a rzeczywiste")
    Selected_atribute = st.selectbox("Wybierz oś atrybut wykresu",selected.columns)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = stones_predicted[Selected_atribute],y=stones_predicted['price'],mode='markers', name='real'))
    fig.add_trace(go.Scatter(x = stones_predicted[Selected_atribute],y=stones_predicted['predicted'],mode='markers', name='predicted '))
    fig.update_layout(
        xaxis_title=Selected_atribute,
        yaxis_title="price",)
    st.plotly_chart(fig)

    st.subheader("współczynnik nacheylenia")
    st.plotly_chart(px.bar(x = X_test.columns[selected_feature_indices2], y = model.coef_,title="selected indices and coefficience", labels={'x':'','y':'price coefficience'}))