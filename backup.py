"""Y_3a = df[["Y/m", "AlBSF slope"]]
Y_3a = pd.concat([Y_3a, X], axis=1)
y_3a = Y_3a.dropna()
y_3 = y_3a["AlBSF slope"]
x_3a = y_3a[[0, 1]]
# print(x_3a)
# print(y_3a.loc[:, [i for i in range(0,2) if isinstance(i, int)]])

# print(y_3a)

Y_4a = df["HJT slope"]
Y_4a = pd.concat([Y_4a, X], axis=1)
y_4a = Y_4a.dropna()
y_4 = y_4a["HJT slope"]
x_4a = y_4a[[0, 1]]

Y_7a = df["BC slope"]
Y_7a = pd.concat([Y_7a, X], axis=1)
y_7a = Y_7a.dropna()
y_7 = y_7a["BC slope"]
x_7a = y_7a[[0, 1]]
# print(Y_3a)

model_3a = sm.OLS(y_3, x_3a)
results_3a = model_3a.fit()
# c3a_hat = c3a_hat.rename(columns={0: 'c3a_hat'})
c3a_a, c3a_b = results_3a.params
c3a_merged = c3a_a + X[1] * c3a_b
# print(c3a_merged)
# print(results_3a.params)

model_4a = sm.OLS(y_4, x_4a)
results_4a = model_4a.fit()
c4a_a, c4a_b = results_4a.params
c4a_merged = c4a_a + X[1] * c4a_b

model_7a = sm.OLS(y_7, x_7a)
results_7a = model_7a.fit()
c7a_a, c7a_b = results_7a.params
c7a_merged = c7a_a + X[1] * c7a_b

# Seasonal Trend 4 parameters
X1 = pd.DataFrame(np.cos((2 * np.pi * t) / 12))
X1 = X1.rename(columns={0: 2})
X2 = pd.DataFrame(np.sin((2 * np.pi * t) / 12))
X2 = X2.rename(columns={0: 3})
X1 = pd.concat([X, X1, X2], axis=1)

Y_3a_s = df[["Y/m", "AlBSF slope"]]
Y_3a_s = pd.concat([Y_3a_s, X1], axis=1)
y_3a_s = Y_3a_s.dropna()
y_3_s = y_3a_s["AlBSF slope"]
x_3a_s = y_3a_s[[0, 1, 2, 3]]

Y_4a_s = df["HJT slope"]
Y_4a_s = pd.concat([Y_4a_s, X1], axis=1)
y_4a_s = Y_4a_s.dropna()
y_4_s = y_4a_s["HJT slope"]
x_4a_s = y_4a_s[[0, 1, 2, 3]]

Y_7a_s = df["BC slope"]
Y_7a_s = pd.concat([Y_7a_s, X1], axis=1)
y_7a_s = Y_7a_s.dropna()
y_7_s = y_7a_s["BC slope"]
x_7a_s = y_7a_s[[0, 1, 2, 3]]

model_3a_s = sm.OLS(y_3_s, x_3a_s)
results_3a_s = model_3a_s.fit()
c3a_a_s, c3a_b_s, c3a_c_s, c3a_d_s = results_3a_s.params
c3a_merged_s = c3a_a_s + X1[1] * c3a_b_s + X1[2] * c3a_c_s + X1[3] * c3a_d_s
# print(results_3a_s.params)

model_4a_s = sm.OLS(y_4_s, x_4a_s)
results_4a_s = model_4a_s.fit()
c4a_a_s, c4a_b_s, c4a_c_s, c4a_d_s = results_4a_s.params
c4a_merged_s = c4a_a_s + X1[1] * c4a_b_s + X1[2] * c4a_c_s + X1[3] * c4a_d_s

model_7a_s = sm.OLS(y_7_s, x_7a_s)
results_7a_s = model_7a_s.fit()
c7a_a_s, c7a_b_s, c7a_c_s, c7a_d_s = results_7a_s.params
c7a_merged_s = c7a_a + X1[1] * c7a_b_s + X1[2] * c7a_c_s + X1[3] * c7a_d_s

# Seasonal Trend no2 2fo
# 2f0 = 1/6
# 3f0 = 36
# 4f0 = 48
X3 = pd.DataFrame(np.cos((2 * np.pi * t) / 6))
X3 = X3.rename(columns={0: 4})
X4 = pd.DataFrame(np.sin((2 * np.pi * t) / 6))
X4 = X4.rename(columns={0: 5})
X4 = pd.concat([X1, X3, X4], axis=1)

Y_3a_s2 = df[["Y/m", "AlBSF slope"]]
Y_3a_s2 = pd.concat([Y_3a_s2, X4], axis=1)
y_3a_s2 = Y_3a_s2.dropna()
y_3_s2 = y_3a_s2["AlBSF slope"]
x_3a_s2 = y_3a_s2[[0, 1, 2, 3, 4, 5]]

Y_4a_s2 = df["HJT slope"]
Y_4a_s2 = pd.concat([Y_4a_s2, X4], axis=1)
y_4a_s2 = Y_4a_s2.dropna()
y_4_s2 = y_4a_s2["HJT slope"]
x_4a_s2 = y_4a_s2[[0, 1, 2, 3, 4, 5]]

Y_7a_s2 = df["BC slope"]
Y_7a_s2 = pd.concat([Y_7a_s2, X4], axis=1)
y_7a_s2 = Y_7a_s2.dropna()
y_7_s2 = y_7a_s2["BC slope"]
x_7a_s2 = y_7a_s2[[0, 1, 2, 3, 4, 5]]

model_3a_s2 = sm.OLS(y_3_s2, x_3a_s2)
results_3a_s2 = model_3a_s2.fit()
c3a_a_s2, c3a_b_s2, c3a_c_s2, c3a_d_s2, c3a_e_s2, c3a_f_s2 = results_3a_s2.params
c3a_merged_s2 = c3a_a_s2 + X4[1] * c3a_b_s2 + X4[2] * c3a_c_s2 + X4[3] * c3a_d_s2 + X4[4] * c3a_e_s2 + X4[5] * c3a_f_s2

model_4a_s2 = sm.OLS(y_4_s2, x_4a_s2)
results_4a_s2 = model_4a_s2.fit()
c4a_a_s2, c4a_b_s2, c4a_c_s2, c4a_d_s2, c4a_e_s2, c4a_f_s2 = results_4a_s2.params
c4a_merged_s2 = c4a_a_s2 + X4[1] * c4a_b_s + X4[2] * c4a_c_s2 + X4[3] * c4a_d_s2 + X4[4] * c4a_e_s2 + X4[5] * c4a_f_s2

model_7a_s2 = sm.OLS(y_7_s2, x_7a_s2)
results_7a_s2 = model_7a_s2.fit()
c7a_a_s2, c7a_b_s2, c7a_c_s2, c7a_d_s2, c7a_e_s2, c7a_f_s2 = results_7a_s2.params
c7a_merged_s2 = c7a_a_s2 + X4[1] * c7a_b_s2 + X4[2] * c7a_c_s2 + X4[3] * c7a_d_s2 + X4[4] * c7a_e_s2 + X4[5] * c7a_f_s2"""