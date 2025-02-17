# Note: this one
import math as ma
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import OrderedDict
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, ShuffleSplit

data = pd.read_pickle('./data.pkl')

# Creating the model
consts_1 = []
consts_2 = []
consts_3 = []

all_months_1 = []
all_months_2 = []
all_months_3 = []

arch_1 = []
arch_2 = []
arch_3 = []

for y in range(2015, 2020):

    d = data[data["Year"] == str(y)]
    d1 = d.drop(columns=["HJT(W/m^2)", "BC(W/m^2)"], axis=1)
    d1 = d1.dropna().reset_index(drop=True)
    # print(d1)
    d2 = d.drop(columns=["AlBSF(W/m^2)", "BC(W/m^2)"], axis=1)
    d2 = d2.dropna().reset_index(drop=True)
    d3 = d.drop(columns=["AlBSF(W/m^2)", "HJT(W/m^2)"], axis=1)
    d3 = d3.dropna().reset_index(drop=True)

    for month in range(1, 13):
        if month < 10:
            m = '0' + str(month)
        else:
            m = str(month)

        if m in d1["Month"].values:
            Y1 = d1[d1["Month"] == m]["AlBSF(W/m^2)"]

            a1 = Y1
            a1 = a1.rename(m + '/' + str(y)[2:4])
            arch_1.append(a1)

            X = d1[d1["Month"] == m]["Irradiance(W/m^2)"]

            model = sm.OLS(Y1, X)

            results = model.fit()

            c = results.params

            consts_1.append(c)

            all_months_1.append(m + '/' + str(y)[2:4])

        if m in d2["Month"].values:
            Y2 = d2[d2["Month"] == m]["HJT(W/m^2)"]
            a2 = Y2
            a2 = a2.rename(m + '/' + str(y)[2:4])
            arch_2.append(a2)

            X1 = d2[d2["Month"] == m]["Irradiance(W/m^2)"]

            model1 = sm.OLS(Y2, X1)
            results1 = model1.fit()

            c1 = results1.params
            consts_2.append(c1)

            all_months_2.append(m + '/' + str(y)[2:4])

        if m in d3["Month"].values:
            Y3 = d3[d3["Month"] == m]["BC(W/m^2)"]
            a3 = Y3
            a3 = a3.rename(m + '/' + str(y)[2:4])
            arch_3.append(a3)

            X2 = d3[d3["Month"] == m]["Irradiance(W/m^2)"]

            model2 = sm.OLS(Y3, X2)
            results2 = model2.fit()

            c2 = results2.params
            consts_3.append(c2)

            all_months_3.append(m + '/' + str(y)[2:4])

# Scatter Plots
dates = ["1/15", "12/19"]
start, end = [datetime.strptime(_, "%m/%y") for _ in dates]
m_y = OrderedDict(((start + timedelta(_)).strftime(r"%m/%y"), None) for _ in range((end - start).days)).keys()
m_y = list(m_y)
all = m_y

ym = pd.DataFrame(all)
ym = ym.rename(columns={0: "Y/m"})
s1 = pd.DataFrame(consts_1)
s1 = pd.concat([pd.DataFrame(all_months_1), s1], axis=1)
s1 = s1.rename(columns={"Irradiance(W/m^2)": "AlBSF slope", 0: "Y/m"})

s2 = pd.DataFrame(consts_2)
s2 = pd.concat([pd.DataFrame(all_months_2), s2], axis=1)
s2 = s2.rename(columns={"Irradiance(W/m^2)": "HJT slope", 0: "Y/m"})
s3 = pd.DataFrame(consts_3)
s3 = pd.concat([pd.DataFrame(all_months_3), s3], axis=1)
s3 = s3.rename(columns={"Irradiance(W/m^2)": "BC slope", 0: "Y/m"})

d_1 = ym.merge(s1, on=["Y/m"], how='left')
d_2 = d_1.merge(s2, on=["Y/m"], how='left')
df = d_2.merge(s3, on=["Y/m"], how='left')

# Read matlab slopes and make the dataframes
d_3a = pd.read_csv('Solar_Irradiance/csv/monthly_slopes_3a.csv')
d_4a = pd.read_csv('Solar_Irradiance/csv/monthly_slopes_4a.csv')
d_7a = pd.read_csv('Solar_Irradiance/csv/monthly_slopes_7a.csv')

m_3a = all_months_1
m_4a = all_months_2
m_7a = all_months_3

m_3a.remove("11/15")
m_3a.remove("12/15")
m_4a.remove("11/15")
m_4a.remove("12/15")
m_7a.remove("11/15")
m_7a.remove("12/15")

d_3a["Y/m"] = m_3a
d_3a.drop(['Year', 'Month'], axis='columns', inplace=True)
d_3a = d_3a.merge(ym, on=["Y/m"], how='right')
d_4a["Y/m"] = m_4a
d_4a.drop(['Year', 'Month'], axis='columns', inplace=True)
d_4a = d_4a.merge(ym, on=["Y/m"], how='right')
d_7a["Y/m"] = m_7a
d_7a.drop(['Year', 'Month'], axis='columns', inplace=True)
d_7a = d_7a.merge(ym, on=["Y/m"], how='right')
# print(d_4a)
# print(d_7a)

# find OLS parameters for estimated slope
t = np.arange(1, 60)
X = sm.add_constant(t)
X = pd.DataFrame(X)
# print(X)


# Generalize the model
# Making our X dataframe
L = 3
X1 = X
m = 2
# print(X1)
for i in range(1, L + 1):
    print(i)
    X12 = pd.DataFrame(np.cos((2 * i * np.pi * t) / 12))
    X22 = pd.DataFrame(np.sin((2 * i * np.pi * t) / 12))
    X12 = X12.rename(columns={0: m})
    X22 = X22.rename(columns={0: m + 1})
    X1 = pd.concat([X1, X12, X22], axis=1)
    m += 2


# Different architectures and their models
columns = [j for j in range(0, m)]
# AlBSF
Y3a = df[["AlBSF slope"]]
Y3a = pd.concat([Y3a, X1], axis=1)
Y3a = Y3a.dropna()
y3a = Y3a["AlBSF slope"]

# Linear Trend Model
x3al = Y3a[[0, 1]]

# Seasonal Trend
# l = 1
x3a = Y3a[columns[:3]]
# l = 2
x3a2 = Y3a[columns[:6]]
# l = 3
x3a3 = Y3a[columns[:8]]


# Models
# Linear
model3al = sm.OLS(y3a, x3al)
results3al = model3al.fit()
c3al = results3al.params
c3alm = c3al[0] + c3al[1] * X[1]

# Seasonal
# l = 1
model3a = sm.OLS(y3a, x3a)
results3a = model3a.fit()
c3a = results3a.params
c3am = c3a[0] + [c3a[i] * X1[i] for i in columns[1:3]]

# l = 2
model3a2 = sm.OLS(y3a, x3a2)
results3a2 = model3a2.fit()
c3a2 = results3a2.params
c3am2 = c3a2[0] + [c3a2[i] * X1[i] for i in columns[1:5]]

# l = 3
model3a3 = sm.OLS(y3a, x3a3)
results3a3 = model3a3.fit()
c3a3 = results3a3.params
c3am3 = c3a3[0] + [c3a3[i] * X1[i] for i in columns[1:]]

# HJT
Y4a = df[["HJT slope"]]
Y4a = pd.concat([Y4a, X1], axis=1)
Y4a = Y4a.dropna()
y4a = Y4a["HJT slope"]

# Linear
x4al = Y4a[[0, 1]]

# Seasonal trend
# l0 = 1
x4a = Y4a[columns[:3]]
# l0 = 2
x4a2 = Y4a[columns[:5]]
# l0 = 3
x4a3 = Y4a[columns]

# Models
# Linear
model4al = sm.OLS(y4a, x4al)
results4al = model4al.fit()
c4al = results4al.params
c4alm = c4al[0] + c4al[1]*X[1]

# Seasonal Trend
# l0 = 1
model4a = sm.OLS(y4a, x4a)
results4a = model4a.fit()
c4a = results4a.params
c4am = c4a[0] + [c4a[i]*X1[i] for i in columns[1:3]]
# l0 = 2
model4a2 = sm.OLS(y4a, x4a2)
results4a2 = model4a2.fit()
c4a2 = results4a2.params
c4am2 = c4a2[0] + [c4a2[i]*X1[i] for i in columns[1:5]]
# l0 = 3
model4a3 = sm.OLS(y4a, x4a3)
results4a3 = model4a3.fit()
c4a3 = results4a3.params
c4am3 = c4a3[0] + [c4a3[i]*X1[i] for i in columns[1:]]

# BC
Y7a = df[["BC slope"]]
Y7a = pd.concat([Y7a, X1], axis=1)
Y7a = Y7a.dropna()
y7a = Y7a["BC slope"]

# Linear
x7al = Y7a[[0, 1]]

# Seasonal Trend
# l0 = 1
x7a = Y7a[columns[:3]]
# l0 = 2
x7a2 = Y7a[columns[:5]]
# l0 = 3
x7a3 = Y7a[columns]

# Models
# Linear
model7al = sm.OLS(y7a, x7al)
results7al = model7al.fit()
c7al = results7al.params
c7alm = c7al[0] + c7al[1]*X1[1]

# Seasonal
# l0 = 1
model7a = sm.OLS(y7a, x7a)
results7a = model7a.fit()
c7a = results7a.params
c7am = c7a[0] + [c7a[i]*X1[i] for i in columns[1:3]]
# l0 = 2
model7a2 = sm.OLS(y7a, x7a2)
results7a2 = model7a2.fit()
c7a2 = results7a2.params
c7am2 = c7a2[0] + [c7a2[i]*X1[i] for i in columns[1:5]]
# l0 = 3
model7a3 = sm.OLS(y7a, x7a3)
results7a3 = model7a3.fit()
c7a3 = results7a3.params
c7am3 = c7a3[0] + [c7a3[i] * X1[i] for i in columns[1:]]

# Cross-Validation
k = 10
rmse_l_3, rmse_3, rmse_3_2, rmse_3_3 = [], [], [], []
mae_l_3, mae_3, mae_3_2, mae_3_3 = [], [], [], []
rmse_l_4, rmse_4, rmse_4_2, rmse_4_3 = [], [], [], []
mae_l_4, mae_4, mae_4_2, mae_4_3 = [], [], [], []
rmse_l_7, rmse_7, rmse_7_2, rmse_7_3 = [], [], [], []
mae_l_7, mae_7, mae_7_2, mae_7_3 = [], [], [], []

for i in range(k):
    print(i)
    # Linear
    # AlBSF
    x_tr3a, x_ts3a, y_tr3a, y_ts3a = train_test_split(x3al, y3a, test_size=0.3, random_state=i**2 + 50)
    ml3a = sm.OLS(y_tr3a, x_tr3a).fit()
    prl3a = ml3a.predict(x_ts3a)
    mse3l = mean_squared_error(y_ts3a, prl3a)
    mae3l = mean_absolute_error(y_ts3a, prl3a)
    mae_l_3.append(mae3l)
    rmse_l_3.append(ma.sqrt(mse3l))

    # HJT
    x_tr4a, x_ts4a, y_tr4a, y_ts4a = train_test_split(x4al, y4a, test_size=0.3, random_state=i ** 2 + 50)
    ml4a = sm.OLS(y_tr4a, x_tr4a).fit()
    prl4a = ml4a.predict(x_ts4a)
    mse4l = mean_squared_error(y_ts4a, prl4a)
    mae4l = mean_absolute_error(y_ts4a, prl4a)
    mae_l_4.append(mae4l)
    rmse_l_4.append(ma.sqrt(mse4l))

    # BC
    x_tr7a, x_ts7a, y_tr7a, y_ts7a = train_test_split(x7al, y7a, test_size=0.3, random_state=i ** 2 + 50)
    ml7a = sm.OLS(y_tr7a, x_tr7a).fit()
    prl7a = ml7a.predict(x_ts7a)
    mse7l = mean_squared_error(y_ts7a, prl7a)
    mae7l = mean_absolute_error(y_ts7a, prl7a)
    mae_l_7.append(mae7l)
    rmse_l_7.append(ma.sqrt(mse7l))

    # Seasonal
    # l0 = 1
    # AlBSF
    xtr3a, xts3a, ytr3a, yts3a = train_test_split(x3a, y3a, test_size=0.3, random_state=i ** 2 + 50)
    m3a = sm.OLS(ytr3a, xtr3a).fit()
    pr3a = m3a.predict(xts3a)
    mse_3 = mean_squared_error(yts3a, pr3a)
    mae3 = mean_absolute_error(yts3a, pr3a)
    mae_3.append(mae3)
    rmse_3.append(ma.sqrt(mse_3))

    # HJT
    xtr4a, xts4a, ytr4a, yts4a = train_test_split(x4a, y4a, test_size=0.3, random_state=i ** 2 + 50)
    m4a = sm.OLS(ytr4a, xtr4a).fit()
    pr4a = m4a.predict(xts4a)
    mse_4 = mean_squared_error(yts4a, pr4a)
    mae4 = mean_absolute_error(yts4a, pr4a)
    mae_4.append(mae4)
    rmse_4.append(ma.sqrt(mse_4))

    # BC
    xtr7a, xts7a, ytr7a, yts7a = train_test_split(x7a, y7a, test_size=0.3, random_state=i ** 2 + 50)
    m7a = sm.OLS(ytr7a, xtr7a).fit()
    pr7a = m7a.predict(xts7a)
    mse_7 = mean_squared_error(yts7a, pr7a)
    mae7 = mean_absolute_error(yts7a, pr7a)
    mae_7.append(mae7)
    rmse_7.append(ma.sqrt(mse_7))

    # l0 = 2
    # AlBSF
    xtr3a2, xts3a2, ytr3a2, yts3a2 = train_test_split(x3a2, y3a, test_size=0.3, random_state=i ** 2 + 50)
    m3a2 = sm.OLS(ytr3a2, xtr3a2).fit()
    pr3a2 = m3a2.predict(xts3a2)
    mse_32 = mean_squared_error(yts3a2, pr3a2)
    mae32 = mean_absolute_error(yts3a2, pr3a2)
    mae_3_2.append(mae32)
    rmse_3_2.append(ma.sqrt(mse_32))

    # HJT
    xtr4a2, xts4a2, ytr4a2, yts4a2 = train_test_split(x4a2, y4a, test_size=0.3, random_state=i ** 2 + 50)
    m4a2 = sm.OLS(ytr4a2, xtr4a2).fit()
    pr4a2 = m4a2.predict(xts4a2)
    mse_42 = mean_squared_error(yts4a2, pr4a2)
    mae42 = mean_absolute_error(yts4a2, pr4a2)
    mae_4_2.append(mae42)
    rmse_4_2.append(ma.sqrt(mse_42))

    # BC
    xtr7a2, xts7a2, ytr7a2, yts7a2 = train_test_split(x7a2, y7a, test_size=0.3, random_state=i ** 2 + 50)
    m7a2 = sm.OLS(ytr7a2, xtr7a2).fit()
    pr7a2 = m7a2.predict(xts7a2)
    mse_72 = mean_squared_error(yts7a2, pr7a2)
    mae72 = mean_absolute_error(yts7a2, pr7a2)
    mae_7_2.append(mae72)
    rmse_7_2.append(ma.sqrt(mse_72))

    # l0 = 3
    # AlBSF
    xtr3a3, xts3a3, ytr3a3, yts3a3 = train_test_split(x3a3, y3a, test_size=0.3, random_state=i ** 2 + 50)
    m3a3 = sm.OLS(ytr3a3, xtr3a3).fit()
    pr3a3 = m3a3.predict(xts3a3)
    mse_33 = mean_squared_error(yts3a3, pr3a3)
    mae33 = mean_absolute_error(yts3a3, pr3a3)
    mae_3_3.append(mae33)
    rmse_3_3.append(ma.sqrt(mse_33))

    # HJT
    xtr4a3, xts4a3, ytr4a3, yts4a3 = train_test_split(x4a3, y4a, test_size=0.3, random_state=i ** 2 + 50)
    m4a3 = sm.OLS(ytr4a3, xtr4a3).fit()
    pr4a3 = m4a3.predict(xts4a3)
    mse_43 = mean_squared_error(yts4a3, pr4a3)
    mae43 = mean_absolute_error(yts4a3, pr4a3)
    mae_4_3.append(mae43)
    rmse_4_3.append(ma.sqrt(mse_43))

    # BC
    xtr7a3, xts7a3, ytr7a3, yts7a3 = train_test_split(x7a3, y7a, test_size=0.3, random_state=i ** 2 + 50)
    m7a3 = sm.OLS(ytr7a3, xtr7a3).fit()
    pr7a3 = m7a3.predict(xts7a3)
    mse_73 = mean_squared_error(yts7a3, pr7a3)
    mae73 = mean_absolute_error(yts7a3, pr7a3)
    mae_7_3.append(mae73)
    rmse_7_3.append(ma.sqrt(mse_73))

rmse = [rmse_l_3, rmse_l_4, rmse_l_7, rmse_3, rmse_4, rmse_7, rmse_3_2, rmse_4_2, rmse_7_2, rmse_3_3]
rmse3 = [rmse_l_3, rmse_3, rmse_3_2, rmse_3_3]
rmse4 = [rmse_l_4, rmse_4, rmse_4_2, rmse_4_3]
rmse7 = [rmse_l_7, rmse_7, rmse_7_2, rmse_7_3]
mae = [mae_l_3, mae_l_4, mae_l_7, mae_3, mae_4, mae_7, mae_3_2, mae_4_2, mae_7_2]
mae3 = [mae_l_3, mae_3, mae_3_2, mae_3_3]
mae4 = [mae_l_4, mae_4, mae_4_2, mae_4_3]
mae7 = [mae_l_7, mae_7, mae_7_2, mae_7_3]
# CV plots
# Linear
# RMSE
# AlBSF
fig = plt.figure(figsize=(10, 7))

# Creating axes instance
ax = fig.add_subplot(111)

# Creating plot
bp = ax.boxplot(rmse3)

# x-axis labels
ax.set_xticklabels(['AlBSF_L', 'AlBSF_l1', 'AlBSF_l2', 'AlBSF_l3'])
ax.set_xlabel('AlBSF')
ax.set_ylabel('RMSE')

# Adding title
plt.title("RMSE")

# show plot
plt.show()

# HJT
fig = plt.figure(figsize=(10, 7))

# Creating axes instance
ax = fig.add_subplot(111)

# Creating plot
bp = ax.boxplot(rmse4)

# x-axis labels
ax.set_xticklabels(['HJT_L', 'HJT_l1', 'HJT_l2', 'HJT_l3'])
ax.set_xlabel('HJT')
ax.set_ylabel('RMSE')

# Adding title
plt.title("RMSE")

# show plot
plt.show()

# BC
fig = plt.figure(figsize=(10, 7))

# Creating axes instance
ax = fig.add_subplot(111)

# Creating plot
bp = ax.boxplot(rmse7)

# x-axis labels
ax.set_xticklabels(['BC_L', 'BC_l1', 'BC_l2', 'BC_l3'])
ax.set_xlabel('BC')
ax.set_ylabel('RMSE')

# Adding title
plt.title("RMSE")

# show plot
plt.show()
#########################################################
# MAE
# AlBSF
fig = plt.figure(figsize=(10, 7))

# Creating axes instance
ax = fig.add_subplot(111)

# Creating plot
bp = ax.boxplot(mae3)

# x-axis labels
ax.set_xticklabels(['AlBSF_L', 'AlBSF_l1', 'AlBSF_l2', 'AlBSF_l3'])
ax.set_xlabel('AlBSF')
ax.set_ylabel('MAE')

# Adding title
plt.title("MAE")

# show plot
plt.show()

# HJT
fig = plt.figure(figsize=(10, 7))

# Creating axes instance
ax = fig.add_subplot(111)

# Creating plot
bp = ax.boxplot(mae4)

# x-axis labels
ax.set_xticklabels(['HJT_L', 'HJT_l1', 'HJT_l2', 'HJT_l3'])
ax.set_xlabel('HJT')
ax.set_ylabel('MAE')

# Adding title
plt.title("MAE")

# show plot
plt.show()

# BC
fig = plt.figure(figsize=(10, 7))

# Creating axes instance
ax = fig.add_subplot(111)

# Creating plot
bp = ax.boxplot(mae7)

# x-axis labels
ax.set_xticklabels(['BC_L', 'BC_l1', 'BC_l2', 'BC_l3'])
ax.set_xlabel('BC')
ax.set_ylabel('MAE')

# Adding title
plt.title("MAE")

# show plot
plt.show()
# Seasonal trend
# Residual Error
"""# Seasonal trend
err_3a = results_3a_s.resid
err_4a = results_4a_s.resid
err_7a = results_7a_s.resid

# Seasonal Trend 2f0
err_3a_2 = results_3a_s2.resid
err_4a_2 = results_4a_s2.resid
err_7a_2 = results_7a_s.resid
# print(err_3a)
mu, std = stats.norm.fit(err_3a_2)
mu4, std4 = stats.norm.fit(err_4a_2)
mu7, std7 = stats.norm.fit(err_7a_2)"""

# Histogram of the residual error
# plt.hist(x=err_3a_2, density=True, linewidth=0, bins='auto', alpha=0.7, rwidth=0.85, label='Residual Error AlBSF 2f0')
# plt.hist(x=err_4a_2, density=True, linewidth=0, bins='auto', alpha=0.7, rwidth=0.85, label='Residual Error HJT 2f0')
# plt.hist(x=err_7a_2, density=True, linewidth=0, bins='auto', alpha=0.7, rwidth=0.85, label='Residual Error BC 2f0')


# Plot the PDF.
"""xmin, xmax = plt.xlim() # the maximum x values from the histogram above
x = np.linspace(xmin, xmax, 100) # generate some x values
p = stats.norm(loc=mu, scale=std).pdf(x)
plt.plot(x, p, color='red', label='Gaussian PDF')
plt.suptitle("Histogram of the Residual Error")
plt.title("PDF of the Guassian Distribution")
plt.xlabel("Residual Error")
plt.ylabel("Frequency")
plt.legend(loc='best')
plt.show()"""

# Estimated slope/efficiency plots
"""fig1, ax = plt.subplots()
ax = plt.gca()
plt.title("Estimated slope for AlBSF")
plt.xlabel("Months")
plt.ylabel("Estimated slope for each month")
plt.scatter(x=df["Y/m"], y=df["AlBSF slope"], marker="*", label="AlBSF")
plt.scatter(x=d_3a["Y/m"], y=d_3a["Slope"], marker="^", label="AlBSF.mat")
plt.plot(df["Y/m"], c3a_merged, label="Linear trend", color='red')
plt.plot(df["Y/m"], c3a_merged_s, label="Seasonal trend", color='green')
plt.plot(df["Y/m"], c3a_merged_s2, label="Seasonal trend 2f0", color='black')
plt.legend(loc='best')
plt.grid(True)
plt.yticks([-0.1, 0, 0.1, 0.2, 0.3])
plt.xticks(ticks=[i for i in range(0, 60, 6)], labels=['1/15', '7/15', '1/16', '7/16', '1/17', '7/17', '1/18', '7/18', '1/19', '7/19'])
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')
plt.show()

fig2, ax1 = plt.subplots()
ax1 = plt.gca()
plt.title("Estimated slope for HJT")
plt.xlabel("Months")
plt.ylabel("Estimated slope for each month")
plt.scatter(x=df["Y/m"], y=df["HJT slope"], marker="^", label="HJT")
plt.scatter(x=d_4a["Y/m"], y=d_4a["Slope"], marker="*", label="HJT.mat")
plt.plot(df["Y/m"], c4a_merged, label="Linear trend", color='red')
plt.plot(df["Y/m"], c4a_merged_s, label="Seasonal trend", color='green')
plt.plot(df["Y/m"], c4a_merged_s2, label="Seasonal trend 2f0", color='black')
plt.legend(loc='best')
plt.grid(True)
plt.yticks([-0.1, 0, 0.1, 0.2, 0.3])
plt.xticks(ticks=[i for i in range(0, 60, 6)], labels=['1/15', '7/15', '1/16', '7/16', '1/17', '7/17', '1/18', '7/18', '1/19', '7/19'])
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')
plt.show()

fig3, ax2 = plt.subplots()
ax2 = plt.gca()
plt.title("Estimated slope for BC")
plt.xlabel("Months")
plt.ylabel("Estimated slope for each month")
plt.scatter(x=df["Y/m"], y=df["BC slope"], marker="p", label="BC")
plt.scatter(x=d_7a["Y/m"], y=d_7a["Slope"], marker="*", label="BC.mat")
plt.plot(df["Y/m"], c7a_merged, label="Linear trend", color='red')
plt.plot(df["Y/m"], c7a_merged_s, label="Seasonal trend", color='green')
plt.plot(df["Y/m"], c7a_merged_s2, label="Seasonal trend 2f0", color='black')
plt.legend(loc='best')
plt.grid(True)
plt.yticks([-0.1, 0, 0.1, 0.2, 0.3])
plt.xticks(ticks=[i for i in range(0, 60, 6)], labels=['1/15', '7/15', '1/16', '7/16', '1/17', '7/17', '1/18', '7/18', '1/19', '7/19'])
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')
plt.show()"""
