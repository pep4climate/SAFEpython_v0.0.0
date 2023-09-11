#%% Step 1: (import python modules)
import os
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

from input_variables import input_variables
from Timber_Beam import Timber_Beam

#%% Step 2: (setup the Hymod model)

# Specify the directory where the data are stored
mydir = os.getcwd()
# Load data:

Joist_length = 4.5;  # [m]
Joist_spacing = 0.3; # [m]

# Select the limit state to consider
LSs=['SLS','ULS'];
LS = 1; # 1 if ULS and 2 for SLS

# Definition of the joist cross-section
# B = [47	47	47	47	47	75	75	75	75	75]/1000; % Joist cross-section typical depths (m)
# H = [150 175 200 225 250 150 175 200 225 250]/1000; % Joist cross-section typical width (m)
Cross_section = 10; # You can pick among 10 cross sections

# Selection of the exposure class and loading duration
#class = 'C1';
#loading = 'Long term';
Kmod_factor = 0.7;
Smod_factor = 0.5;

[eta_f, beta_f, eta_Emean, beta_Emean, eta_Gmean, beta_Gmean, rho_mean, sigma_rho, eta_snow, beta_snow] = input_variables();
# eta_f: median bending strength
# beta_f: log standard deviation of the bending strength
# eta_Emean: mean elastic modulus
# beta_Emean: log standard deviation of the elastic modulus
# eta_Gmean: mean shear elastic modulus
# beta_Gmean: log standard deviation of the shear elastic modulus
# rho_mean: mean self weight
# sigma_rho: standard deviation self weight
# eta_snow: median snow load on the roof
# beta_snow: log standard deviation of snow load on the roof 

# Define inputs:
# Parameters =    f      E      G     rho     q
#DistrFuns    = {'logn','logn','logn','norm','logn'} ; % Parameter distribution
#DistrPar  = { [ log(eta_f) beta_f ]; [ log(eta_Emean) beta_Emean ]; [ log(eta_Gmean) beta_Gmean ]; [ rho_mean sigma_rho ] ; [ eta_snow beta_snow ] } ; % Parameter ranges (from literature)
#x_labels = {'f','E','G','rho','q'} ;
# Define output:
#myfun = 'Timber_Beam' ;

# Number of uncertain parameters subject to SA:
M = 5

# Parameter distributions:
distr_fun=[st.lognorm,st.lognorm,st.lognorm,st.norm,st.lognorm]
distr_par= [[beta_f, eta_f], [beta_Emean, eta_Emean], [beta_Gmean, eta_Gmean], [sigma_rho, rho_mean], [beta_snow, eta_snow]];

# Name of parameters (will be used to customize plots):
X_Labels = ['f','E','G','rho','q']

# Define output:
fun_test = Timber_Beam


#%% Step 3 (sample inputs space)
samp_strat = 'lhs' # Latin Hypercube
N = 3000  #  Number of samples
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)


#%% Step 4 (run the model)
Y = model_execution(fun_test, X, Joist_length,Joist_spacing,Cross_section,Smod_factor,Kmod_factor)

#%% Step 5a (Regional Sensitivity Analysis with threshold)

# Visualize input/output samples (this may help finding a reasonable value for
# the output threshold):
plt.figure()
pf.scatter_plots(X, Y[:, 0], Y_Label='rmse', X_Labels=X_Labels)
plt.ylim([0, 2])
plt.show()
plt.figure()
pf.scatter_plots(X, Y[:, 1], Y_Label='bias', X_Labels=X_Labels)
plt.show()

# Set output threshold:
rmse_thres = 1   # threshold for the first obj. fun.
bias_thres = 1   # behavioural threshold for the second obj. fun.

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

