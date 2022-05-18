import streamlit as st
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
from scipy.spatial import distance
import numpy as np
from random import randint
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from variables import ubi  as ubi_clientes
from variables import ubistores  as ubi_stores


COSTO_IMP_TIENDA=[50.03,57.53,46.28,58.79]
NUMERO_DIAS = 31
# Costo por kilometro en pesos reales (Brasil).
COSTO_KM = 6.62
# Capacidad diaría de tiendas.
CAPACIDA_DIA_TIENDA = [100,150,50,100]
SCHEDULESCLIENTES = [[0,6],[1,6],[10,10],[11,12],[13,6]]


st.title('Entrega de TC Rio de Janeiro')

@st.cache
def load_data(clients,stores):
    ubi = np.array(clients)
    ubistores = np.array(stores)
    # Cálculo de número de clientes como número de ubicaciones.
    numero_clientes=len(ubi)
    # Cálculo de número de tiendas como número de ubicaciones.
    longitud_tienda = len(COSTO_IMP_TIENDA)
    # Cálculo de distancias entre los clientes y las tiendas de repartición.
    distancia_tienda_cliente = [[distance.euclidean(ubi[k], ubistores[i]) for i in range(longitud_tienda)] for k in range(numero_clientes)]
    # Planteamiento del Solver para respuestas enteras.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    # Variables que indican la tienda, el día de reparto y el cliente.
    x = [[[solver.IntVar(0, 1, f'x[{tienda}, {dia},{cliente} ]') for cliente in range(numero_clientes) ] for dia in range(NUMERO_DIAS) ] for tienda  in range(len(COSTO_IMP_TIENDA))]
    # Definición de función objetivo.
    objetivo = solver.Objective()
    for i in range(longitud_tienda):
      for j in range(NUMERO_DIAS):
        for k in range(numero_clientes):
          objetivo.SetCoefficient(x[i][j][k], COSTO_IMP_TIENDA[i])
    for i in range(longitud_tienda):
        for j in range(NUMERO_DIAS):
            for k in range(numero_clientes):
                objetivo.SetCoefficient(x[i][j][k], COSTO_KM*distancia_tienda_cliente[k][i])
    objetivo.SetMinimization()
    # Restricción.
    for j in range(NUMERO_DIAS):  
        solver.Add(solver.Sum([x[i][j][k] for i in range(longitud_tienda) for k in range(numero_clientes)]) <= CAPACIDA_DIA_TIENDA[i])
    # Restricción.
    for k in range(numero_clientes):
        solver.Add(solver.Sum([x[i][j][k] for i in range(longitud_tienda) for j in range(NUMERO_DIAS)])==1)
    for cliente in SCHEDULESCLIENTES:
        solver.Add(solver.Sum([x[i][cliente[1]][cliente[0]] for i in range(longitud_tienda)])==1)
    # Restricción
    for j in range(NUMERO_DIAS):
        for i in range(longitud_tienda):
            solver.Add(solver.Sum([x[i][j][k] for k in range(numero_clientes)])>=1)
    status= solver.Solve()
    dataset=[]
    if status == pywraplp.Solver.OPTIMAL:
        print('Solución')
        print('Función objetivo =', solver.Objective().Value())
        for i in range(longitud_tienda):
            for j in range(NUMERO_DIAS):
                for k in range(numero_clientes):
                    if x[i][j][k].solution_value() != 0.0:
                        #print(x[i][j][k])
                        dataset.append([i,j,k]) 
    else:
        print('Error solución imposible')
    data = pd.DataFrame(dataset, index = None, columns=None, dtype='category')
    data.columns=['OriginStore','day','client']
    data['count'] = 1

    data = data.sort_values('client')
    data['index']= data['client']
    data.set_index('index',inplace = True)

    dfubi = pd.DataFrame (ubi, index = None, columns=None)
    dfubi.columns=['lat','log']
    dfubi['name']=1

    data = pd.concat([data,dfubi], axis = 1)
    return data

data_load_state = st.text('Loading data...')
data = load_data(ubi_clientes, ubi_stores)
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

fig = px.bar(data, x='day', y='count',
              color='OriginStore',
              hover_data=['client'],
            #  facet_row="OriginStore",
             color_discrete_sequence=px.colors.qualitative.Safe,
             labels={'OriginStore':'Origin Store'}, height=400)

st.plotly_chart(fig, use_container_width=True)

if st.checkbox('All days'):
    st.subheader('Map')
    figubi = px.scatter_mapbox(data, lat="lat", lon="log", hover_name="name", hover_data=["OriginStore"], 
                            zoom=13, height=500, color = 'OriginStore')
    figubi.update_layout(mapbox_style="open-street-map")
    figubi.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(figubi, use_container_width=True)

category = st.selectbox(
     'Choise day',
     (data['day'].unique()))

figubi = px.scatter_mapbox(data[data.day==category], lat="lat", lon="log", hover_name="name", hover_data=["OriginStore"], 
                         zoom=13, height=500, color = 'OriginStore')
figubi.update_layout(mapbox_style="open-street-map")
figubi.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(figubi, use_container_width=True)
