import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Estimacion del precio de la gasolina en funcion de estado, mes y año ''')
st.image("gasolinera.png", caption="Gasolinera.",width="100")
st.image("tabla.png", caption="Tabla de equivalencia de estados.")

st.header('Datos de referncia')

def user_input_features():
  # Entrada
  Año = st.number_input('Año(yyyy):',  min_value=1982, max_value=2100, value = 2020, step = 1)
  Mes_index = st.number_input('Mes(Mm):', min_value=0, max_value=11, value = 0, step = 1)
  Estado_index = st.number_input('Estado (0-31):', min_value=0, max_value=230, value = 0, step = 1)

  user_input_data = {'Año': Año,
                     'Mes_index': Mes_index,
                     'Estado_index': Estado_index}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

gasolina =  pd.read_csv('PrecioGasolina.csv')
X = gasolina.drop(columns='Valor')
y = gasolina['Valor']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df.Año + b1[1]*df.Mes_index + b1[2]*df.Estado_index

st.subheader('Cálculo del costo de la gasolina')
st.write('La cantidad precicha para las variables elgidas:', prediccion)
