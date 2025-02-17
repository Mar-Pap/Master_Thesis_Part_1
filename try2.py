# Note: This one
import pandas as pd
import numpy as np
import os
from statistics import mean, stdev
from pathlib import Path

from matplotlib import pyplot as plt

dir = 'Solar_Irradiance/data/Irradiance/'
dir_list = os.listdir(dir)

prev = pd.DataFrame({})
print("Irradiance")
for y in range(2015, 2021):
    # print("Year", y)
    for m in range(1, 13):
        # print("Month", m)
        if m < 10:
            dir_irr = dir + str(y) + '/' + str(y) + '_0' + str(m) + '.txt'

        else:
            dir_irr = dir + str(y) + '/' + str(y) + '_' + str(m) + '.txt'

        if os.path.exists(dir_irr):
            d = pd.read_csv(dir_irr, delim_whitespace=True)
            columns1 = d.columns
            d1 = d[[columns1[0]]]
            d2 = d[[columns1[1]]]
            # print(d1)
            d[["Year", "Month", "Day"]] = d[columns1[0]].str.split("-", expand=True)
            d.drop(columns1[0], axis='columns', inplace=True)  # OK
            d[["Hour", "Minutes"]] = d[columns1[1]].str.split(":", expand=True)
            d.drop(columns1[1], axis="columns", inplace=True)  # OK
            update = d["Hour"].values + ":00"

            if prev.empty:
                prev = d
            else:
                d = pd.concat([prev, d], axis=0)
                prev = d
        else:
            print("Month", m, "of year", y, "doesn't exist")

data = d.loc[:, ["Year", "Month", "Day", "Hour", "Minutes", "Irradiance(W/m^2)"]]
dir_ey = 'Solar_Irradiance/data/EnergyYield/'

print("Energy Yield")
prev_3a = pd.DataFrame({})
prev_4a = pd.DataFrame({})
prev_7a = pd.DataFrame({})
for y in range(2015, 2021):
    print("Year", y)
    for m in range(1, 13):
        print("Month", m)
        if m < 10:
            dir_ey_3a = dir_ey + str(y) + '/' + str(y) + '_0' + str(m) + '_Arch3a.txt'
            dir_ey_4a = dir_ey + str(y) + '/' + str(y) + '_0' + str(m) + '_Arch4a.txt'
            dir_ey_7a = dir_ey + str(y) + '/' + str(y) + '_0' + str(m) + '_Arch7a.txt'
        else:
            dir_ey_3a = dir_ey + str(y) + '/' + str(y) + '_' + str(m) + '_Arch3a.txt'
            dir_ey_4a = dir_ey + str(y) + '/' + str(y) + '_' + str(m) + '_Arch4a.txt'
            dir_ey_7a = dir_ey + str(y) + '/' + str(y) + '_' + str(m) + '_Arch7a.txt'


        if os.path.exists(dir_ey_3a):
            d_3a = pd.read_csv(dir_ey_3a, delim_whitespace=True)

            columns1 = d_3a.columns
            d1 = d_3a[[columns1[0]]]
            d2 = d_3a[[columns1[1]]]

            d_3a[["Year", "Month", "Day"]] = d_3a[columns1[0]].str.split("-", expand=True)
            d_3a.drop(columns1[0], axis='columns', inplace=True)  # OK
            d_3a[["Hour", "Minutes"]] = d_3a[columns1[1]].str.split(":", expand=True)
            d_3a.drop(columns1[1], axis="columns", inplace=True)  # OK
            update = d_3a["Hour"].values + ":00"
            if prev_3a.empty:
                prev_3a = d_3a
            else:
                d_3a = pd.concat([prev_3a, d_3a], axis=0)
                prev_3a = d_3a
        else:
            print("Month", m, "of year", y, "doesn't exist for the method ALBSF")

        if os.path.exists(dir_ey_4a):
            d_4a = pd.read_csv(dir_ey_4a, delim_whitespace=True)
            # print(d)
            columns1 = d_4a.columns
            d1 = d_4a[[columns1[0]]]
            d2 = d_4a[[columns1[1]]]
            # print(d1)
            d_4a[["Year", "Month", "Day"]] = d_4a[columns1[0]].str.split("-", expand=True)
            d_4a.drop(columns1[0], axis='columns', inplace=True)  # OK
            d_4a[["Hour", "Minutes"]] = d_4a[columns1[1]].str.split(":", expand=True)
            d_4a.drop(columns1[1], axis="columns", inplace=True)  # OK
            update = d_4a["Hour"].values + ":00"

            if prev_4a.empty:
                prev_4a = d_4a
            else:
                d_4a = pd.concat([prev_4a, d_4a], axis=0)
                prev_4a = d_4a

        else:
            print("Month", m, "of year", y, "doesn't exist for the method HJT")

        if os.path.exists(dir_ey_7a):
            d_7a = pd.read_csv(dir_ey_7a, delim_whitespace=True)
            # print(d)
            columns1 = d_7a.columns
            d1 = d_7a[[columns1[0]]]
            d2 = d_7a[[columns1[1]]]
            # print(d1)
            d_7a[["Year", "Month", "Day"]] = d_7a[columns1[0]].str.split("-", expand=True)
            d_7a.drop(columns1[0], axis='columns', inplace=True)  # OK
            d_7a[["Hour", "Minutes"]] = d_7a[columns1[1]].str.split(":", expand=True)
            d_7a.drop(columns1[1], axis="columns", inplace=True)  # OK
            update = d_7a["Hour"].values + ":00"

            if prev_7a.empty:
                prev_7a = d_7a
            else:
                d_7a = pd.concat([prev_7a, d_7a], axis=0)
                prev_7a = d_7a
        else:
            print("Month", m, "of year", y, "doesn't exist for the method BC")


# merging the data
d_merg3 = data.merge(d_3a, on=['Year', 'Month', 'Day', 'Hour', 'Minutes'], how='left')
# print(d_merg3)
d_merg4 = d_merg3.merge(d_4a, on=['Year', 'Month', 'Day', 'Hour', 'Minutes'], how='left')
# print(d_merg4)
data_merged = d_merg4.merge(d_7a, on=['Year', 'Month', 'Day', 'Hour', 'Minutes'], how='left')
# print(data_merged)
# Find the indeces with null
# print(np.where(d_merg3['AlBSF(W/m^2)'].isnull()))

# print(data_merged.columns)

# Task 1
data_merged["HJT(W/m^2)"] = np.where((data_merged["Year"] == '2019') & (data_merged["Month"] == '07'), np.nan, data_merged["HJT(W/m^2)"])
data_merged["BC(W/m^2)"] = np.where((data_merged["Year"] == '2019') & (data_merged["Month"] == '07'), np.nan, data_merged["BC(W/m^2)"])

# Task 2
# print(np.where(data_merged["Irradiance(W/m^2)"] < 10**(-2)))
data_merged["Irradiance(W/m^2)"] = np.where(data_merged["Irradiance(W/m^2)"] < 10**(-2), np.nan, data_merged["Irradiance(W/m^2)"])
data_merged["AlBSF(W/m^2)"] = np.where(data_merged["AlBSF(W/m^2)"] < 10**(-2), np.nan, data_merged["AlBSF(W/m^2)"])
data_merged["HJT(W/m^2)"] = np.where(data_merged["HJT(W/m^2)"] < 10**(-2), np.nan, data_merged["HJT(W/m^2)"])
data_merged["BC(W/m^2)"] = np.where(data_merged["BC(W/m^2)"] < 10**(-2), np.nan, data_merged["BC(W/m^2)"])

# Task 3
data_merged["AlBSF(W/m^2)"] = np.where((data_merged["AlBSF(W/m^2)"] < 10) & (data_merged["Irradiance(W/m^2)"] > 200), np.nan, data_merged["AlBSF(W/m^2)"])
data_merged["HJT(W/m^2)"] = np.where((data_merged["HJT(W/m^2)"] < 10) & (data_merged["Irradiance(W/m^2)"] > 200), np.nan, data_merged["HJT(W/m^2)"])
data_merged["BC(W/m^2)"] = np.where((data_merged["BC(W/m^2)"] < 10) & (data_merged["Irradiance(W/m^2)"] > 200), np.nan, data_merged["BC(W/m^2)"])


# Task 4
data_merged["Irradiance(W/m^2)"] = np.where(data_merged["Irradiance(W/m^2)"] > 1100, np.nan, data_merged["Irradiance(W/m^2)"])


columns = data_merged.columns
# dm = data_merged.to_records()
data_merged.to_pickle('./data.pkl')


# print(dm)
# np.save('Solar_Irradiance/data/data_merged.npy', dm)

"""filepath = Path('Solar_Irradiance/data/data_merged.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
data_merged.to_csv(filepath)"""




# Plots
"""
# 1. figure
columns = data_merged.columns
irrad = data_merged[columns[:6]]
m_irr = []
y_err_up = []
y_err_down = []
no_month = int(input("For which month should I plot? "))
# print(no_month)
years = input("Should I plot for all the years or for a specific year? ")
# print(years)
for i in range(1, 2):
    if i < 10:
        m = "0" + str(i)
    else:
        m = str(i)

if no_month < 10:
    m = "0" + str(no_month)
else:
    m = str(no_month)

irr = irrad[irrad["Month"] == m]

if years != "all":
    irr = irr[irr["Year"] == str(years)]

# print(irr) OK

for h in range(6, 19):
    if h < 10:
        h1 = "0" + str(h)
    else:
        h1 = str(h)
    # print(irr[irr["Hour"] == h1])
    mean_irr = mean(irr[irr["Hour"] == h1]["Irradiance(W/m^2)"])
    print(mean_irr)
    m_irr.append(mean_irr)
    print(stdev(irr[irr["Hour"] == h1]["Irradiance(W/m^2)"], xbar=mean_irr))
    y_err_up.append(mean_irr + stdev(irr[irr["Hour"] == h1]["Irradiance(W/m^2)"], xbar=mean_irr))
    y_err_down.append(mean_irr - stdev(irr[irr["Hour"] == h1]["Irradiance(W/m^2)"], xbar=mean_irr))
    # print(mean(irr["Irradiance(W/m^2)"]))
    days = irrad[irrad["Day"] == "01"]
    mean_irr = mean(days)
    # print(mean_irr)
# print(irrad[irrad["Month"] == "01"])

y_err = [y_err_down, y_err_up]
# print(m_irr)
x = [i for i in range(0, 13)]
# y_err = np.linspace(min(m_irr), max(m_irr), len(x))
plt.plot(x, m_irr)
plt.errorbar(x, m_irr, yerr=y_err)
plt.show()"""


# Scatter plots
"""print(data_merged.columns)
plt.title("Arch7a")
plt.xlabel("Irradiance")
plt.ylabel("Energy Yield")
plt.scatter(data_merged["Irradiance(W/m^2)"], data_merged["BC(W/m^2)"])
plt.show()"""

