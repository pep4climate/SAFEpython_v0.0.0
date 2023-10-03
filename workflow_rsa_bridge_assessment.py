#%% Step 1: (import python modules)
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd

import SAFEpython.RSA_thres as Rt # module to perform RSA with threshold
import SAFEpython.RSA_groups as Rg # module to perform RSA with groups
import SAFEpython.plot_functions as pf # module to visualize the results
from SAFEpython.model_execution import model_execution # module to execute the model
from SAFEpython.sampling import AAT_sampling # module to perform the input sampling
from SAFEpython.util import aggregate_boot  # function to aggregate the bootstrap results

from input_variables import input_variables
from Bridge_Assessment import bridge_assessment

#%% Step 2: (setup the Hymod model)

# Specify the directory where the data are stored
# mydir = os.getcwd()
# Load data:

# Define inputs:
# Parameters =    uw

# [eta, beta] = input_variables()
# eta_Wmean: log median of the masonry unit weight
# beta_Wmean: log standard deviation of the masonry unit weight

# [shape, scale] = input_variables()
[median_Wmean, beta_Wmean, median_Phimean, beta_Phimean] = input_variables()

# Parameter distributions:
distr_fun = [st.lognorm, st.lognorm]

# distr_par = [[eta, beta]]
# print("etaWmean is: ", eta, " beta_Wmean is: ", beta)
distr_par = [[beta_Wmean, median_Wmean], [beta_Phimean, median_Phimean]] # pay attention in inputting lognormal parameters in this order
print(distr_par)

# Number of uncertain parameters subject to SA:
M = len(distr_fun)

# Name of parameters (will be used to customize plots):
X_Labels = ['uw', 'phi_prime']

# Define output:
fun_test = bridge_assessment

#%% Step 3 (sample inputs space)
samp_strat = 'rsu' # 'lhs' for Latin Hypercube # 'rsu' for Random Uniform
N = 10  #  Number of samples
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)
print(np.sort(X)) 
print(np.log(np.sort(X)))
print(X[0])
# print(X.ndim)
# print(X.shape)
# print(X.size)
# print(type(X))

# X=[[10], [22], [100]] # fake sampling
# X_array = np.array(X)
# X_array = np.log(np.array(X))
# print("X_array is: ", X_array)

plt.hist(np.sort(X), bins='auto')
plt.show() # without the command plt.show(), no plot is shown

df_X = pd.DataFrame(X)
file_name_X = 'realisations.csv'
df_X.to_csv(file_name_X, index=False)

# %% Step 4 (run the model)
# Y = fun_test(X) # input X realisations into the bridge_assessment function
# Y = model_execution(fun_test, X)
csv_file = r"C:\Program Files\LimitState\GEO3.6\bin\solution_file_SAFE_5.csv"
df = pd.read_csv(csv_file)
column_name = 'Answer'
adequacy_factor = df[column_name]
print("The Adequacy Factor is: ", adequacy_factor)

Y = adequacy_factor.to_numpy()
print("Y is:", Y)

df_Y = pd.DataFrame(Y)
file_name_Y = 'adequacy_factor.csv'
df.to_csv(file_name_Y, index = False)

#%% Step 5a (Regional Sensitivity Analysis with threshold)

# Visualize input/output samples (this may help finding a reasonable value for
# the output threshold):
plt.figure()
label_of_Y = 'Capacity/Demand'
# pf.scatter_plots(X, Y[:, 0], Y_Label=label_of_Y, X_Labels=X_Labels)
pf.scatter_plots(X, Y, Y_Label=label_of_Y, X_Labels=X_Labels)
# plt.ylim([0, 2])
plt.show()
# plt.figure()
# pf.scatter_plots(X, Y[:, 1], Y_Label='bias', X_Labels=X_Labels)
# plt.show()

# Set output threshold:
rmse_thres = 1   # threshold for the first obj. fun.
bias_thres = 1   # behavioural threshold for the second obj. fun.
# rmse_thres = np.ones((len(Y),1))
# bias_thres = rmse_thres

# RSA (find behavioural parameterizations, i.e. parameterizations with
# output values below the threshold):
# threshold = [rmse_thres, bias_thres]
# threshold = [rmse_thres, bias_thres, rmse_thres, bias_thres, rmse_thres, bias_thres, rmse_thres, bias_thres, rmse_thres, bias_thres,]
threshold = [rmse_thres]
mvd, spread, irr, idxb = Rt.RSA_indices_thres(X, Y, threshold)

# Highlight the behavioural parameterizations in the scatter plots:
plt.figure()
pf.scatter_plots(X, Y, Y_Label=label_of_Y , X_Labels=X_Labels, idx=idxb)
plt.show()

# Plot parameter CDFs with legend:
Rt.RSA_plot_thres(X, idxb, X_Labels=X_Labels, str_legend=['behav', 'non-behav'])
plt.show()

# Check the ranges of behavioural parameterizations by parallel coordinate plot:
plt.figure()
pf.parcoor(X, X_Labels=X_Labels, idx=idxb)
plt.show()

# Plot the sensitivity indices (maximum vertical distance between
# parameters CDFs):
plt.figure()
pf.boxplot1(mvd, X_Labels=X_Labels, Y_Label='mvd')
plt.show()

# Compute sensitivity indices with confidence intervals using bootstrapping
Nboot = 1000
# Warning: the following line may take some time to run, as the computation of
# CDFs is costly:
mvd, spread, irr, idxb = Rt.RSA_indices_thres(X, Y, threshold, Nboot=Nboot)
# mvd, spread and irr have shape (Nboot, M)

# Compute mean and confidence intervals of the sensitivity indices (mvd,
# maximum vertical distance) across the bootstrap resamples:
mvd_m, mvd_lb, mvd_ub = aggregate_boot(mvd) # shape (M,)
# Plot results:
plt.figure()
pf.boxplot1(mvd_m, X_Labels=X_Labels, Y_Label='mvd', S_lb=mvd_lb, S_ub=mvd_ub)
plt.show()

# Repeat computations using an increasing number of samples to assess
# convergence:
NN = np.linspace(N/5, N, 5).astype(int)
mvd, spread, irr = Rt.RSA_convergence_thres(X, Y, NN, threshold)
# Plot the sensitivity measures (maximum vertical distance between
# parameters CDFs) as a function of the number of samples:
plt.figure()
pf.plot_convergence(mvd, NN, X_Label='no of samples', Y_Label='mvd', labelinput=X_Labels)
plt.show()

# Repeat convergence analysis using bootstrapping to derive confidence bounds:
Nboot = 1000
# Warning: the following line may take some time to run, as the computation of
# CDFs is costly:
mvd, spread, irr = Rt.RSA_convergence_thres(X, Y, NN, threshold, Nboot=Nboot)
# mvd, spread and irr have shape (Nboot, M)

# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
mvd_m, mvd_lb, mvd_ub = aggregate_boot(mvd) # shape (R,M)
# Plot results:
plt.figure()
pf.plot_convergence(mvd_m, NN, mvd_lb, mvd_ub, X_Label='no of samples',
                    Y_Label='mvd', labelinput=X_Labels)
plt.show()

