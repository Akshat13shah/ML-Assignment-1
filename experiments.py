# %%
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 5  # Number of times to run each experiment to calculate the average values
# %%

# %%
# Function to create fake data (take inspiration from usage.py)
#RIRO
t_1 = []
t_2 = []
t_1_1 = []
t_2_1 = []
i = 0
for N in range (50,60):
    for M in range (3,6):
        t_1_1_1 = []
        t_1_1_2 = []
        for j in range(num_average_time):
            X = pd.DataFrame(np.random.randn(N, M))
            y = pd.Series(np.random.randn(N))
            criteria = "mse"
            s = time.time()
            tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
            tree.fit(X, y)
            t_1_1_1.append(time.time()-s)
            y_hat = tree.predict(X)
            t_1_1_2.append(time.time()-s)
            #tree.plot()
            #print("Criteria :", criteria)
            #print("RMSE: ", rmse(y_hat, y))
            #print("MAE: ", mae(y_hat, y))
            i = i+1 
            print(i)
        t_1.append(np.mean(t_1_1_1))
        t_2.append(np.mean(t_1_1_2))

    t_1_1.append(np.mean(t_1))
    t_2_1.append(np.mean(t_2))
print(len(t_1),len(t_2))


# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ...
N = np.linspace(50,59, num = 10)
M = np.linspace(3,5, num = 3)
t_1 = np.array(t_1)
t_2 = np.array(t_2)
N_mesh, M_mesh = np.meshgrid(N,M)
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.scatter(N_mesh.flatten(),M_mesh.flatten(),t_1.flatten())
ax.set(xlabel='Time', ylabel='No. of Samples', title='Fitting Time')
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.scatter(N_mesh.flatten(),M_mesh.flatten(),t_2.flatten())
ax.set(xlabel='Time', ylabel='No. of Samples', title='Total Time')
plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(projection = "3d")
#ax.scatter(N_mesh.flatten(),M_mesh.flatten(),t_2.flatten())
# plt.show()
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots
plt.plot(N, t_1_1, label = "Fitting Time")
plt.plot(N, t_2_1, label = "Total Time")
plt.xlabel('No. of columns')
plt.ylabel('Time')
plt.legend()
plt.show()
# %%

# %%
#RIDO
t_1 = []
t_2 = []
t_1_1 = []
t_2_1 = []
i = 0
for N in range (50,60):
    for M in range (3,6):
        t_1_1_1 = []
        t_1_1_2 = []
        for j in range(num_average_time):
            X = pd.DataFrame(np.random.randn(N, M))
            y = pd.Series(np.random.randint(M, size=N), dtype="category")
            criteria = "entropy"
            s = time.time()
            tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
            tree.fit(X, y)
            t_1_1_1.append(time.time()-s)
            y_hat = tree.predict(X)
            t_1_1_2.append(time.time()-s)
            #tree.plot()
            #print("Criteria :", criteria)
            #print("RMSE: ", rmse(y_hat, y))
            #print("MAE: ", mae(y_hat, y))
            i = i+1 
            print(i)
        t_1.append(np.mean(t_1_1_1))
        t_2.append(np.mean(t_1_1_2))

    t_1_1.append(np.mean(t_1))
    t_2_1.append(np.mean(t_2))
print(len(t_1),len(t_2))


# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ...
N = np.linspace(50,59, num = 10)
M = np.linspace(3,5, num = 3)
t_1 = np.array(t_1)
t_2 = np.array(t_2)
N_mesh, M_mesh = np.meshgrid(N,M)
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.scatter(N_mesh.flatten(),M_mesh.flatten(),t_1.flatten())
ax.set(xlabel='Time', ylabel='No. of Samples', title='Fitting Time')
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.scatter(N_mesh.flatten(),M_mesh.flatten(),t_2.flatten())
ax.set(xlabel='Time', ylabel='No. of Samples', title='Total Time')
plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(projection = "3d")
#ax.scatter(N_mesh.flatten(),M_mesh.flatten(),t_2.flatten())
# plt.show()
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots
plt.plot(N, t_1_1, label = "Fitting Time")
plt.plot(N, t_2_1, label = "Total Time")
plt.xlabel('No. of columns')
plt.ylabel('Time')
plt.legend()
plt.show()
# %%

# %%
#DIRO
t_1 = []
t_2 = []
t_1_1 = []
t_2_1 = []
i = 0
for N in range (50,60):
    for M in range (3,6):
        t_1_1_1 = []
        t_1_1_2 = []
        for j in range(num_average_time):
            X = pd.DataFrame({i: pd.Series(np.random.randint(10, size=N), dtype="category") for i in range(M)})
            y = pd.Series(np.random.randn(N))
            criteria = "mse"
            s = time.time()
            tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
            tree.fit(X, y)
            t_1_1_1.append(time.time()-s)
            y_hat = tree.predict(X)
            t_1_1_2.append(time.time()-s)
            #tree.plot()
            #print("Criteria :", criteria)
            #print("RMSE: ", rmse(y_hat, y))
            #print("MAE: ", mae(y_hat, y))
            i = i+1 
            print(i)
        t_1.append(np.mean(t_1_1_1))
        t_2.append(np.mean(t_1_1_2))

    t_1_1.append(np.mean(t_1))
    t_2_1.append(np.mean(t_2))
print(len(t_1),len(t_2))


# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ...
N = np.linspace(50,59, num = 10)
M = np.linspace(3,5, num = 3)
t_1 = np.array(t_1)
t_2 = np.array(t_2)
N_mesh, M_mesh = np.meshgrid(N,M)
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.scatter(N_mesh.flatten(),M_mesh.flatten(),t_1.flatten())
ax.set(xlabel='Time', ylabel='No. of Samples', title='Fitting Time')
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.scatter(N_mesh.flatten(),M_mesh.flatten(),t_2.flatten())
ax.set(xlabel='Time', ylabel='No. of Samples', title='Total Time')
plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(projection = "3d")
#ax.scatter(N_mesh.flatten(),M_mesh.flatten(),t_2.flatten())
# plt.show()
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots
plt.plot(N, t_1_1, label = "Fitting Time")
plt.plot(N, t_2_1, label = "Total Time")
plt.xlabel('No. of columns')
plt.ylabel('Time')
plt.legend()
plt.show()
# %%

# %%
#DIDO
t_1 = []
t_2 = []
t_1_1 = []
t_2_1 = []
i = 0
for N in range (50,60):
    for M in range (3,6):
        t_1_1_1 = []
        t_1_1_2 = []
        for j in range(num_average_time):
            X = pd.DataFrame({i: pd.Series(np.random.randint(10, size=N), dtype="category") for i in range(M)})
            y = pd.Series(np.random.randint(10, size=N), dtype="category")
            criteria = "entropy"
            s = time.time()
            tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
            tree.fit(X, y)
            t_1_1_1.append(time.time()-s)
            y_hat = tree.predict(X)
            t_1_1_2.append(time.time()-s)
            #tree.plot()
            #print("Criteria :", criteria)
            #print("RMSE: ", rmse(y_hat, y))
            #print("MAE: ", mae(y_hat, y))
            i = i+1 
            print(i)
        t_1.append(np.mean(t_1_1_1))
        t_2.append(np.mean(t_1_1_2))

    t_1_1.append(np.mean(t_1))
    t_2_1.append(np.mean(t_2))
print(len(t_1),len(t_2))


# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ...
N = np.linspace(50,59, num = 10)
M = np.linspace(3,5, num = 3)
t_1 = np.array(t_1)
t_2 = np.array(t_2)
N_mesh, M_mesh = np.meshgrid(N,M)
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.scatter(N_mesh.flatten(),M_mesh.flatten(),t_1.flatten())
ax.set(xlabel='Time', ylabel='No. of Samples', title='Fitting Time')
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.scatter(N_mesh.flatten(),M_mesh.flatten(),t_2.flatten())
ax.set(xlabel='Time', ylabel='No. of Samples', title='Total Time')
plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(projection = "3d")
#ax.scatter(N_mesh.flatten(),M_mesh.flatten(),t_2.flatten())
# plt.show()
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots
plt.plot(N, t_1_1, label = "Fitting Time")
plt.plot(N, t_2_1, label = "Total Time")
plt.xlabel('No. of columns')
plt.ylabel('Time')
plt.legend()
plt.show()
# %%