import pandas as pd
import streamlit as st
import random
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import svm

#basic things
data = pd.read_csv("supply_chain_data.csv")
data
data['A'] = range(1, 101)
data.set_index('A', inplace=True)
data
#(data.info())
#(data.describe())
#(data.isnull().sum())
#(data.shape)


#(data.columns.tolist())
#Product_type = (data['Product type'])
#Product_type
#####################################################
#calculer le montant total des ventes pour chaque SKU
#####################################################
for i in data:
 if i  == 'Product type':
  data['Tot_Vente_SKU'] = (data['Price']) * (data['Number of products sold'])

(data['Tot_Vente_SKU'])
##########################
#montant totale des ventes
##########################
tot_vente = sum(data['Tot_Vente_SKU'])
tot_vente
#montant total des ventes pour chaque produit
haircare = data["Product type"] == "haircare"
filtered_product_haircare = data[haircare]
filtered_product_haircare
filtered_product_haircare.loc[:,'tot_haircare'] = filtered_product_haircare["Price"] * filtered_product_haircare["Number of products sold"]
filtered_product_haircare.loc[:,'tot_haircare']
tot_haircare = filtered_product_haircare['tot_haircare'].sum()
tot_haircare

skincare = data["Product type"] == "skincare"
filtered_product_skincare = data[skincare]
filtered_product_skincare
filtered_product_skincare.loc[:,'tot_skincare'] = filtered_product_skincare["Price"] * filtered_product_skincare["Number of products sold"]
filtered_product_skincare.loc[:,'tot_skincare']
tot_skincare = filtered_product_skincare['tot_skincare'].sum()
tot_skincare

cosmetics = data["Product type"] == "cosmetics"
filtered_product_cosmetics = data[cosmetics]
filtered_product_cosmetics
filtered_product_cosmetics.loc[:,'tot_cosmetics'] = filtered_product_cosmetics["Price"] * filtered_product_cosmetics["Number of products sold"]
filtered_product_cosmetics.loc[:,'tot_cosmetics']
tot_cosmetics = filtered_product_cosmetics['tot_cosmetics'].sum()
tot_cosmetics
###########################################
# The percent %
###########################################
haircare_in_percent = tot_haircare * 100/tot_vente
haircare_in_percent
skincare_in_percent = tot_skincare * 100/tot_vente
skincare_in_percent
cosmetics_in_percent = tot_cosmetics * 100/tot_vente
cosmetics_in_percent
# for the pie
display = ["Transportation modes", "Shipping costs"]
ship = data[display]
ship 
total_shipping_costs = ship.groupby("Transportation modes")["Shipping costs"].sum().reset_index()
total_shipping_costs
###########
# The title
###########
st.title("SUPPLY CHAIN ANALYSIS 🚢 ✈ 🚛 🚉 ")
####################################
# we start to plot gauge and barchart
####################################
col1, col2, col3 = st.columns(3)

#Gauge haircare
def plot_gauge():
 fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = 632896,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = "TOTAL HAIRCARE SALES ",
    gauge = {
      'axis': {'range': [0, 2285549.96]},
      'bar': {'color': "red"},
      'steps': [
                {'range': [0, 500000], 'color': 'whitesmoke'},
                {'range': [500000, 2285549.96], 'color': 'tomato'}
       ],
      }
      ))
 fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10, pad=8))

 fig.show()
 st.plotly_chart(fig, use_container_width=True)

with col1:
  plot_gauge()

#Gauge skincare
def plot_gauge():
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=1052073,  
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "TOTAL SKINCARE SALES "},
        gauge={
            'axis': {'range': [0, 2000000]},  # Set the range of the gauge
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 1000000], 'color': 'lightgray'},
                {'range': [1000000, 2000000], 'color': 'blue'}
            ],
        }
    ))
    
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10, pad=8))
    
    # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Call the function to plot the gauge

with col2:
  plot_gauge()

#Gauge cosmetics
def plot_gauge():
 fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = 600580,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = "TOTAL COSMETICS SALES",
    gauge = {
      'axis': {'range': [0, 2285549.96]},
      'bar': {'color': "green"},
      'steps': [
                {'range': [0, 500000], 'color': 'whitesmoke'},
                {'range': [500000, 2285549.96], 'color': 'lime'}
       ],
      }
      ))
 fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10, pad=8))

 fig.show()
 st.plotly_chart(fig, use_container_width=True)
#

with col3:
  plot_gauge()

colA, colB = st.columns(2)
#####################################################################################
#st.title('Analyse des Ventes et des Coûts d\'Expédition')
part_de_marche_produit = data.groupby('Product type')['Revenue generated'].sum().reset_index()

fig1 = px.pie(part_de_marche_produit, 
               names='Product type', 
               values='Revenue generated', 
               title='Market Share by Product Type',
               labels={'Product type': 'Type de Produit', 'Revenue generated': 'Revenu Généré'})

with colA:
  st.plotly_chart(fig1)

repartition_transporteur = data.groupby('Shipping carriers')['Shipping costs'].sum().reset_index()

fig2 = px.pie(repartition_transporteur, 
               names='Shipping carriers', 
               values='Shipping costs', 
               title='Distribution of Shipping Costs by Carrier',
               labels={'Shipping carriers': 'Transporteur', 'Shipping costs': 'Coûts d\'Expédition'})

with colB:
  st.plotly_chart(fig2)

cola, colb, colc = st.columns(3)
#barchart AMOUNT PRICE VS PRODUCT TYPE
def plot_bottom_left():
  fig = px.histogram(data,
                    x="Product type",
                    y = "Price",
                    title=' PRICE VS PRODUCT TYPE',
                    color='Product type',
                    color_discrete_sequence=px.colors.qualitative.Set1
                    )
  fig.show()
  st.plotly_chart(fig, use_container_width=True)

with cola:
  plot_bottom_left()


#######################################
# PIE CHART
#######################################

def plot_pie():
  fig = px.pie(total_shipping_costs, 
            values='Shipping costs',
            names='Transportation modes',
            title= 'SHIPPING COSTS VS TRANSPORTATION MODES (💸 vs ✈)',
            color_discrete_sequence=px.colors.sequential.RdBu)
  return fig

fig = plot_pie()
fig.show()
with colc:
  st.plotly_chart(fig, use_container_width=True)

ventes_par_demographie = data.groupby('Customer demographics')['Revenue generated'].sum().reset_index()

# Créer le graphique en barres pour les ventes par démographie avec Plotly
fig2 = px.bar(ventes_par_demographie, 
               x='Customer demographics', 
               y='Revenue generated', 
               title='Sales by Customer Demographics',
               labels={'Customer demographics': 'Démographie des Clients', 'Revenue generated': 'Revenu Généré'},
               text='Revenue generated',
               color_discrete_sequence=px.colors.sequential.RdBu)

with colb:
  st.plotly_chart(fig2)
  
  
st.title('Inventory Optimization')
st.write(data.head())

# Calculer la demande annuelle (en supposant que le nombre de produits vendus est une bonne approximation)
data['Annual Demand'] = data['Number of products sold'] * 12  # Estimation sur 12 mois

# Coût par commande (en utilisant Shipping costs comme approximation)
data['Cost per Order'] = data['Shipping costs']

# Coût de stockage par unité (en utilisant Manufacturing costs comme approximation)
data['Holding Cost per Unit'] = data['Manufacturing costs']

# Calculer la quantité de commande économique (EOQ)
def calculate_eoq(demand, order_cost, holding_cost):
    if holding_cost > 0:  # Éviter la division par zéro
        return np.sqrt((2 * demand * order_cost) / holding_cost)
    else:
        return 0

data['EOQ'] = data.apply(lambda row: calculate_eoq(row['Annual Demand'], row['Cost per Order'], row['Holding Cost per Unit']), axis=1)

# Afficher les résultats
st.subheader('Economic Order Quantity (EOQ)')
st.write(data[['Product type', 'SKU', 'EOQ']])

# Visualisation des résultats
fig = px.bar(data, 
            x='Product type', 
            y='EOQ', 
            title='Economic Order Quantity (EOQ) by Product Type',
            labels={'EOQ': 'Economic Order Quantity', 'Product type': 'Product type'},
            color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(fig)
###################################################
# for the heatmap
##################################################
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder


Le = LabelEncoder()
#data['Product type'] = Le.fit_transform(data['Product type'])
data['Customer demographics'] = Le.fit_transform(data['Customer demographics'])
data['Supplier name'] = Le.fit_transform(data['Supplier name'])
data['Location'] = Le.fit_transform(data['Location'])
data['Inspection results'] = Le.fit_transform(data['Inspection results'])
data['Routes'] = Le.fit_transform(data['Routes'])
#data['Shipping carriers'] = Le.fit_transform(data['Shipping carriers'])
data['SKU'] = Le.fit_transform(data['SKU'])
data['Transportation modes'] = Le.fit_transform(data['Transportation modes'])
data

def map_plot():
  heatmap_data = data.pivot_table(values='Revenue generated', 
                                  index='Manufacturing lead time', 
                                  )
  plt.figure(figsize=(8, 6))
  sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlGnBu')
  plt.title('Heatmap Revenue generated prorata au Manufacturing lead time')
  plt.show()
 
map_plot()


st.pyplot()


#######################################################
# ML part
#######################################################





data['Revenue'] = data['Price'] * data['Number of products sold']
data['Lead time'] = pd.to_timedelta(data['Lead time'])

Le = LabelEncoder()
data['SKU'] = Le.fit_transform(data['SKU'])
data

features = ['SKU', 'Price', 'Availability', 'Stock levels', 'Lead times', 'Shipping costs','Order quantities']
target = 'Revenue'

X = data[features]
y = data[target]

X = pd.get_dummies(X, columns=['SKU'], drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
(f'Erreur absolue moyenne: {mae}')
("R squared", metrics.r2_score(y_test,y_pred))
("MSE", mean_squared_error(y_test,y_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse
medae = median_absolute_error(y_test, y_pred)
(f"Le Median Absolute Error (MedAE) est : {medae}")





