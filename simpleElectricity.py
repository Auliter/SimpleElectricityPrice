import requests
import pandas as pd
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math

urlDemand = (
    "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn?settlementDateFrom=2024-10-07&settlementDateTo=2024-10-11&settlementPeriod=15&settlementPeriod=16&settlementPeriod=17&settlementPeriod=18&settlementPeriod=19&settlementPeriod=20&settlementPeriod=21&settlementPeriod=22&settlementPeriod=23&settlementPeriod=24&settlementPeriod=25&settlementPeriod=26&settlementPeriod=27&settlementPeriod=28&settlementPeriod=29&settlementPeriod=30&settlementPeriod=31&settlementPeriod=32&settlementPeriod=33&settlementPeriod=34&settlementPeriod=35&settlementPeriod=36&settlementPeriod=37&settlementPeriod=38&settlementPeriod=39&settlementPeriod=40&format=json"
)
response = requests.get(urlDemand)
data = json.loads(response.text)
dfDemand = pd.json_normalize(data['data'])
sortedDemand = dfDemand.sort_values(by='startTime')
finalDemandData = sortedDemand[['startTime','initialTransmissionSystemDemandOutturn']].copy()

urlPrice = ("https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index?from=2024-10-07T00%3A00Z&to=2024-10-11T00%3A00Z&settlementPeriodFrom=15&settlementPeriodTo=40&dataProviders=APXMIDP&format=json")
response2 = requests.get(urlPrice)
dataprice = json.loads(response2.text)
dfPrice = pd.json_normalize(dataprice['data'])
filteredPrice = dfPrice[(dfPrice['settlementPeriod'] >= 15) & (dfPrice['settlementPeriod'] <= 40)]
sortedPrice = filteredPrice.sort_values(by='startTime')
finalPrice = sortedPrice[['startTime','price']].copy()

merged = pd.merge(finalDemandData, finalPrice, on="startTime", suffixes=("_x", "_y"))
X = merged["initialTransmissionSystemDemandOutturn"].values.reshape(-1,1)
y = merged["price"].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

##estimate the noise term
residual = y_pred - y
std = np.std(residual)
mean = sum(residual) / len(residual)
print(var,std)
