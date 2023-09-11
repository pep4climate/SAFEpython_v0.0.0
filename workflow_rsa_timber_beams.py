"""
This script provides an application example of Regional Sensitivity
Analysis (RSA)

METHODS



#%% Step 1: (import python modules)

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import SAFEpython.RSA_thres as Rt # module to perform RSA with threshold
import SAFEpython.RSA_groups as Rg # module to perform RSA with groups
import SAFEpython.plot_functions as pf # module to visualize the results
from SAFEpython.model_execution import model_execution # module to execute the model
from SAFEpython.sampling import AAT_sampling # module to perform the input sampling
from SAFEpython.util import aggregate_boot  # function to aggregate the bootstrap results

from SAFEpython import HyMod

#%% Step 2: (setup the Hymod model)

# Specify the directory where the data are stored
mydir = r'Y:\Home\sarrazin\SAFE\SAFEpython_v0.0.0\data'
# Load data:
data = np.genfromtxt(mydir +'\LeafCatch.txt', comments='%')
rain = data[0:365, 0] # 1-year simulation
evap = data[0:365, 1]
flow = data[0:365, 2]
warmup = 30 # Model warmup period (days)

# Number of uncertain parameters subject to SA:
M = 5

# Parameter ranges (from Kollat et al.(2012)):
xmin = [0, 0, 0, 0, 0.1]
xmax = [400, 2, 1, 0.1, 1]

# Parameter distributions:
distr_fun = st.uniform # uniform distribution
# The shape parameters of the uniform distribution are the lower limit and the
# difference between lower and upper limits:
distr_par = [np.nan] * M
for i in range(M):
    distr_par[i] = [xmin[i], xmax[i] - xmin[i]]

# Name of parameters (will be used to customize plots):
X_Labels = ['Sm', 'beta', 'alfa', 'Rs', 'Rf']

# Define output:
fun_test = HyMod.hymod_MulObj


#%% Step 3 (sample inputs space)
samp_strat = 'lhs' # Latin Hypercube
N = 3000  #  Number of samples
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)

#%% Step 4 (run the model)
Y = model_execution(fun_test, X, rain, evap, flow, warmup)

#%% Step 5a (Regional Sensitivity Analysis with threshold)

# Visualize input/output samples (this may help finding a reasonable value for
# the output threshold):
plt.figure()
pf.scatter_plots(X, Y[:, 0], Y_Label='rmse', X_Labels=X_Labels)
plt.show()
plt.figure()
pf.scatter_plots(X, Y[:, 1], Y_Label='bias', X_Labels=X_Labels)
plt.show()

# Set output threshold:
rmse_thres = 4   # threshold for the first obj. fun.
bias_thres = 0.5 # behavioural threshold for the second obj. fun.

# RSA (find behavioural parameterizations, i.e. parameterizations with
# output values below the threshold):
threshold = [rmse_thres, bias_thres]
mvd, spread, irr, idxb = Rt.RSA_indices_thres(X, Y, threshold)

# Highlight the behavioural parameterizations in the scatter plots:
plt.figure()
pf.scatter_plots(X, Y[:, 0], Y_Label='rmse', X_Labels=X_Labels, idx=idxb)
plt.show()
plt.figure()
pf.scatter_plots(X, Y[:, 1], Y_Label='bias', X_Labels=X_Labels, idx=idxb)
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

#%% Step 5b (Regional Sensitivity Analysis with groups)

mvd_median, mvd_mean, mvd_max, spread_median, spread_mean, spread_max, idx, Yk = \
       Rg.RSA_indices_groups(X, Y[:, 0])

# Plot parameter CDFs:
Rg.RSA_plot_groups(X, idx, Yk)
plt.show()
# Customize labels and legend:
Rg.RSA_plot_groups(X, idx, Yk, X_Labels=X_Labels, legend_title='rmse')
plt.show()

# Parallel coordinate plot with groups:
# (Warning: when using a large number of groups, it may be difficult to interpret
# the parallel plot as the the lines for the different groups may overlap)
plt.figure()
pf.parcoor(X, X_Labels=X_Labels, idx=idx)
plt.show()
