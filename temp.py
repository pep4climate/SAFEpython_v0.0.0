# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#==============================================================================
# 0. Set current working directory and import python modules
#==============================================================================

from __future__ import division, absolute_import, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.io as sio

# Module to perform Regional Sensitivity Analysis:
import SAFEpython.RSA_thres as RSA_tr
# Module to perform Regional Sensitivity Analysis based on grouping:
import SAFEpython.RSA_groups as RSA_gp
# Module to perform Variance-Based Sensitivity Analysis:
import SAFEpython.VBSA as VB
# Module to visualize the results:
import SAFEpython.plot_functions as pf
# Module to execute the model
from SAFEpython.model_execution import model_execution
# Functions to perform the input sampling
from SAFEpython.sampling import AAT_sampling, AAT_sampling_extend
# Function to aggregate results across bootstrap resamples and to calculate RMSE:
from SAFEpython.util import aggregate_boot, RMSE
# Module that simulates the HyMod model:
from SAFEpython import HyMod

# Import the additional function BIAS in the BIAS.py module
# Set path where BIAS.py is saved (CHANGE IT TO YOUR OWM PATH BEFORE RUNNING!!)
path = r'C:\Users\rd14186\OneDrive - University of Bristol\Desktop\SAFEpython_v0.0.0\SAFEpython'
os.chdir(path)
from BIAS import BIAS

#%%
#==============================================================================
# 1. Model setup and One-At-a-Time Sensitivity Analysis
#==============================================================================

# Specify the directory where the data are stored 
# (CHANGE IT TO YOUR OWM PATH BEFORE RUNNING!!)
mydir = os.getcwd()

# Load data and plot the data (one year of daily observations of rainfall, 
# potential evaporation and flow):

# Load data:
data = np.genfromtxt(mydir +'\LeafCatch.txt', comments='%')
rain = data[0:365, 0] # select the first year of data
evap = data[0:365, 1]
flow = data[0:365, 2]
warmup = 30 # Model warmup period (days)

# Plot data:
plt.figure()
plt.subplot(311); plt.plot(rain); plt.ylabel('rainfall (mm/day)')
plt.subplot(312); plt.plot(evap); plt.ylabel('evaporation (mm/day)')
plt.subplot(313); plt.plot(flow); plt.ylabel('flow (mm/day)')
plt.xlabel('time (days)')
plt.show()

# Set the parameters to some tentative values, run the model and plot the 
# resulting streamflow time series:

# Set a tentative parameterization:
param = np.array([200, 0.5, 0.7, 0.05, 0.6]) # Sm (mm), beta (-), alfa (-), Rs (-), Rf (-)
# Run simulation:
flow_sim, _, _ = HyMod.hymod_sim(param, rain, evap)
# Plot results:
plt.figure()
plt.plot(flow, color=[0.7, 0.7, 0.7])
plt.plot(flow_sim, 'k')
plt.ylabel('flow (mm/day)')
plt.xlabel('time (days)')
plt.legend(['obs', 'sim'])
plt.show()

# TO DO:
# One-At-a-Time (OAT) effect: change one parameter (e.g. alfa) and repeat the
# previous steps.
# - What is the effect of varying this parameter (and the others)?
# - Which parameter controls which characteristic (timing, peak, recession
#   phase, etc.) of the simulated flow time series?
# - Which parameters seem to mostly influence the output?
# - What are the pros and cons of OAT sensitivity analysis?

#%%
#==============================================================================
# 2. Visual and qualitative Global Sensitivity Analysis
#==============================================================================

# In this section, we run Monte Carlo (MC) simulations of the model against a
# certain number of input (parameter) samples. Each model simulation provides a
# times series of runoff predictions. To measure the accuracy of each of these
# times series with respect to observations, we define two aggregate output
# metrics to be used for the subsequent steps:
# - the Root Mean Squared Error (RMSE) of the streamflow predictions
# - the volumetric Bias (BIAS) of the streamflow predictions

# Input (parameter) sampling (this involves a number of choices that will be 
# assessed and revised later on):
    
# Define input variability space:
X_Labels = ['Sm', 'beta', 'alfa', 'Rs', 'Rf'] # Name of parameters (used
# to customize plots)
M = len(X_Labels) # Number of parameters
distr_fun = st.uniform # Parameter distributions
xmin = [0, 0, 0, 0, 0.1] # Parameter ranges (lower bound)
xmax = [400, 2, 1, 0.1, 1] # Parameter ranges (upper bound)
# The shape parameters of the uniform distribution are the lower limit and the
# difference between lower and upper limits:
distr_par = [np.nan] * M
for i in range(M):
    distr_par[i] = [xmin[i], xmax[i] - xmin[i]]

# Choose sampling strategy and size:
samp_strat = 'lhs' # sampling strategy
# options:
# 'lhs' = Latin Hypercube sampling
# 'rsu' = Random uniform sampling
N = 150 # Choose the number of samples

# Perform sampling:
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)
# If you want to see what the sample 'X' looks like:
print(X[0:10, :]) # Print to screen the first 10 samples

# Execute the model against all the input samples in 'X':
QQ = model_execution(HyMod.hymod_sim, X, rain, evap)

# Plot Monte Carlo (MC) simulations results and compare with data:
plt.figure()
plt.plot(flow, color=[0.7, 0.7, 0.7]) # plot for legend
plt.plot(np.transpose(QQ), 'k');
plt.plot(flow, color=[0.7, 0.7, 0.7])
plt.ylabel('flow (mm/day)'); plt.xlabel('time (days)')
plt.legend(['obs', 'sim'])
plt.title("Model execution with N = %d" % N)
plt.show()

# Aggregate time series into various scalar output metric(s):
YY = np.nan * np.ones((N, 2))
YY[:, 0] = RMSE(QQ[:, warmup:365], flow[warmup:365])
YY[:, 1] = BIAS(QQ[:, warmup:365], flow[warmup:365])

# If you want to see what the samples in 'YY' looks like:
print(YY.shape) # Check the shape of 'YY'
print(YY[0:10, :]) # Print to screen the first 10 samples

# Select the output metric to be analysed (uncomment option you want to view):
Y = YY[:, 0]; Y_Label = 'RMSE' # RMSE
# Y = YY[:, 1]; Y_Label = 'BIAS' # BIAS

# Scatter plots of the output metric against input samples:
plt.figure()
pf.scatter_plots(X, Y, Y_Label=Y_Label, X_Labels=X_Labels)
plt.title("Scatter plots with N = %d" % N + ', ' + Y_Label, loc='right')
plt.show()

# TO DO:
# - From these scatter plots, which parameter would you say is most influential?
#   Why?
# - Are there parameters that are not influential at all?

#%%
#==============================================================================
# 3. Regional Sensitivity Analysis
#==============================================================================

# In this section, we formally assess the sensitivity of the output metrics
# (RMSE and BIAS) to the model parameters through the Regional Sensitivity
# Analysis (RSA) method.

# Define the threshold:
threshold = 3.5
# Use the function RSA_indices_thres to split into behavioural (Y<threshold)
# and non-behavioural (Y>threshold) sample:
mvd, _, _, idxb = RSA_tr.RSA_indices_thres(X, Y, threshold)
# To learn what the function 'RSA_indices_thres' does, type:
# help(RSA_tr.RSA_indices_thres)
# idxb: indices of behavioural samples

# Plot behavioural MC simulations:
QQb = QQ[idxb, :]
plt.figure()
plt.plot(flow, color=[0.7, 0.7, 0.7]) # plot for legend
plt.plot(np.transpose(QQb), 'r');
plt.plot(flow, color=[0.7, 0.7, 0.7])
plt.ylabel('flow (mm/day)'); plt.xlabel('time (days)')
plt.legend(['obs', 'sim'])
plt.title("Behavioural simulations with N = %d" % N + ', ' + Y_Label + " = %2.2f" % threshold)
plt.show()

# Replot the results with the function `scatter_plots` highlighting the
# behavioural parameterizations:
plt.figure()
pf.scatter_plots(X, Y, Y_Label=Y_Label, X_Labels=X_Labels, idx=idxb)
plt.title("Scatter plots with behavioural simulations with N = %d" % N + ', ' + Y_Label  + " = %2.2f" % threshold, \
          loc='right')
plt.show()
# (red = behavioural, blue = non-behavioural)

# Plot CDFs of behavioural and non-behavioural input samples:
RSA_tr.RSA_plot_thres(X, idxb, X_Labels=X_Labels)
plt.title("CDFs with N = %d" % N + ', ' + Y_Label + " = %2.2f" % threshold, \
          loc='right')
plt.show()

# TO DO:
# - From the CDF plots, which parameter would you say are the most influential?
#   Why?
# - Are these results consistent with the visual analysis of the scatter plots?

# Check the value of KS statistic maximum vertical distance between the two
# CDFs of the inputs):
print(mvd)

# Plot the KS statistic:
plt.figure()
pf.boxplot1(mvd, X_Labels=X_Labels, Y_Label='KS statistic')
plt.title("KS with N = %d" % N + ', ' + Y_Label + " = %2.2f" % threshold)
plt.show()

# TO DO:
# - Are the KS values consistent with the visual inspection of the CDF plots?
# - Did you get the same KS values as the other participants? Why?

#==============================================================================
# 3.1 Assess the effect of the change in threshold
#==============================================================================

# TO DO: Choose a different threshold (L311) and re-run the RSA analysis.
# - How have the results changed by changing the threshold of the output metric?

#==============================================================================
# 3.2 Assess the effect of the definition of output metric
#==============================================================================

# TO DO: Choose a different output metric (L287-288) and re-run the analysis.
# - Does the answer change depending on the performance metric chosen? Why?

#==============================================================================
# 3.3 Assess the robustness of the results (i.e. are the results sample
# independent?)
#==============================================================================

# To assess the robustness of the sensitivity indices, bootstrapping is performed
# (here `Nboot = 1000`):
Y = YY[:, 0]; Y_Label = 'RMSE' # RMSE
# Y = YY[:,1]; Y_Label = 'BIAS' # BIAS
# Define the threshold:
threshold = 3.5

Nboot = 1000 # Number of resamples used for bootstrapping
# Assess sensitivity indices with bootstrapping (WARNING: it may take some time
# to run this line):
mvd, _, _, idxb = RSA_tr.RSA_indices_thres(X, Y, threshold, Nboot=Nboot)
# mvd has shape (Nboot, M)

# Compute mean and confidence intervals of the sensitivity indices (mvd,
# maximum vertical distance or KS statistics) across the bootstrap resamples:
alfa = 0.05 # Significance level for the confidence intervals estimated by
# bootstrapping (default value is 0.05)
mvd_m, mvd_lb, mvd_ub = aggregate_boot(mvd, alfa=alfa) # shape (M,)

# The sensitivity indices with their 95% confidence intervals are plotted:
plt.figure()
pf.boxplot1(mvd_m, S_lb=mvd_lb, S_ub=mvd_ub, X_Labels=X_Labels, \
            Y_Label='KS statistic')
plt.title("KS with 95%% CI, N = %d" % N + ", " + Y_Label + " = %2.2f" % threshold)
plt.show()

# TO DO:
# - Are the sensitivity indices adequately estimated? 
# - Is the sample size large enough?

# Try now to increase the sample size (N) and re-run the analysis:

# Choose the number of samples to be added:
N_new = 2000 # size of new samples
N2 = N_new + N # total sample size

# Add new samples:
X_N2 = AAT_sampling_extend(X, distr_fun, distr_par, N2)
# AAT_sampling_extend allows to extend an existing sample by choosing additional
# samples that maximise the spread in the input space
# X_N2 is the extended sample (it includes the already evaluated samples 'X'
# and the new ones)
# Check the shape is X_N2:
print(X_N2.shape)

# Extract new samples:
X_new = X_N2[N:N2, :]
# Check the shape of X_new:
print(X_new.shape)    

# Compute the output metrics with new sample size:

# Execute the model against the new inputs samples (WARNING: it may take
# some time to run this line):
QQ_new = model_execution(HyMod.hymod_sim, X_new, rain, evap)

# Aggregate time series into various scalar output metric(s):
YY_new = np.nan * np.ones((N_new, 2))
YY_new[:, 0] = RMSE(QQ_new[:, warmup:365], flow[warmup:365])
YY_new[:, 1] = BIAS(QQ_new[:, warmup:365], flow[warmup:365])

# Combine old and new output samples:
YY_N2 = np.concatenate((YY, YY_new))

# Select the output metric to be analysed (uncomment option you want to view):
Y_N2 = YY_N2[:, 0]; Y_Label = 'RMSE' # RMSE
# Y_N2 = YY_N2[:, 1]; Y_Label = 'BIAS' # BIAS

# Use the function RSA_indices_thres to split into behavioural (Y<threshold)
# and non-behavioural (Y>threshold) sample and calculate sensitivity indices
# with bootstrapping (WARNING: it may take some time to run this line):
mvd_N2, _, _, idxb_N2 = RSA_tr.RSA_indices_thres(X_N2, Y_N2, threshold, Nboot=Nboot)
# mvd_N2 has shape (Nboot, M)

# Plot CDFs of behavioural and non-behavioural input samples:
RSA_tr.RSA_plot_thres(X_N2, idxb_N2, X_Labels=X_Labels)
plt.title("CDFs with N = %d" % N2 + ', ' + Y_Label + " = %2.2f" % threshold, \
          loc='right')
plt.show()

# Compute mean and confidence intervals of the sensitivity indices (mvd,
# maximum vertical distance or KS statistics) across the bootstrap resamples:
mvd_N2_m, mvd_N2_lb, mvd_N2_ub = aggregate_boot(mvd_N2, alfa=alfa) # shape (M,)

# Check the bootstrap mean values of the KS statistic:
print(mvd_N2_m)

# Plot the sensitivity indices with their 95% confidence intervals:
plt.figure()
pf.boxplot1(mvd_N2_m, S_lb=mvd_N2_lb, S_ub=mvd_N2_ub, X_Labels=X_Labels, \
            Y_Label='KS statistic')
plt.title("KS with 95%% CI, N = %d" % N2 + ', ' + Y_Label + " = %2.2f" % threshold)
plt.show()

# TO DO:
# - Has the ranking of the parameters changed?
# - Is the sample size large enough now?
# - How do the KS values you obtained compare to the results of the other participants?

# Repeat computations using an increasing number of samples to assess
# convergence:
NN = np.array([150, 300, 500, 1000, 1500, 2150]) # increasing sample sizes
Nboot = 100 # Number of resamples used for bootstrapping (suggested value 
# is 100 for the purpose of this exercise to  limit computational time, but 
# the recommended value for a rigorous analysis is 1000)
mvd_cvg, _, _ = RSA_tr.RSA_convergence_thres(X_N2, Y_N2, NN, threshold, Nboot=Nboot)
# mvd, spread and irr have shape (Nboot, M)

# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
mvd_cvg_m, mvd_cvg_lb, mvd_cvg_ub = aggregate_boot(mvd_cvg) # shape (R,M)
# Plot the sensitivity measures as a function of the number of samples:
plt.figure()
pf.plot_convergence(mvd_cvg_m, NN, mvd_cvg_lb, mvd_cvg_ub, X_Label='no of samples',
                    Y_Label='KS statistic', labelinput=X_Labels)
plt.show()

# TO DO:
# - Has convergence been reached? Why?

#==============================================================================
# 3.4 Assess the effect of the definition of the input factors' space of
#     variability
#==============================================================================

# In this section we assess the effect of the input (parameter) ranges.
# This requires new model executions.

# Define new input factors' ranges:
xmin2 = [0, 0, 0, 0, 0.4] # Parameter ranges (lower bound)
xmax2 = [800, 4, 0.5, 0.4, 1] # Parameter ranges (upper bound)
#xmin = [0, 0, 0, 0, 0.1] # Parameter ranges (lower bound)
#xmax = [400, 2, 1, 0.1, 1] # Parameter ranges (upper bound)

# The shape parameters of the uniform distribution are the lower limit and the
# difference between lower and upper limits:
distr_par2 = [np.nan] * M
for i in range(M):
    distr_par2[i] = [xmin2[i], xmax2[i] - xmin2[i]]

# Sample input factors' space using the larger sample size N2:
X_P2 = AAT_sampling(samp_strat, M, distr_fun, distr_par2, N2)

# Execute the model:
QQ_P2 = model_execution(HyMod.hymod_sim, X_P2, rain, evap)

# Aggregate time series into various scalar output metric(s):
YY_P2 = np.nan * np.ones((N2, 2))
YY_P2[:, 0] = RMSE(QQ_P2[:, warmup:365], flow[warmup:365])
YY_P2[:, 1] = BIAS(QQ_P2[:, warmup:365], flow[warmup:365])

# Select the output metric to be analysed (uncomment option you want to view):
Y_P2 = YY_P2[:, 0]; Y_Label = 'RMSE' # RMSE
# Y_P2 = YY_P2[:, 1]; Y_Label = 'BIAS' # BIAS

# Define the threshold:
threshold = 3.5


# Use the function RSA_indices_thres to split into behavioural (Y<threshold)
# and non-behavioural (Y>threshold) sample and calculate sensitivity indices
# with bootstrapping (WARNING: it may take some time to run this line):
mvd_P2, _, _, idxb_P2 = RSA_tr.RSA_indices_thres(X_P2, Y_P2, threshold, Nboot=Nboot)
# mvd_N2 has shape (Nboot, M)

# Visualize the scatter plots:
# Old parameterisation:
plt.figure()
pf.scatter_plots(X_N2, Y_N2, Y_Label=Y_Label, X_Labels=X_Labels, idx=idxb_N2)
plt.title("Scatter plots with old parameterisation with N = %d" % N2 + ', ' + Y_Label  + " = %2.2f" % threshold, \
          loc='right')
plt.show()
# New parameterisation   : 
plt.figure()
pf.scatter_plots(X_P2, Y_P2, Y_Label=Y_Label, X_Labels=X_Labels, idx=idxb_P2)
plt.title("Scatter plots with new parameterisation with N = %d" % N2 + ', ' + Y_Label  + " = %2.2f" % threshold, \
          loc='right')
plt.show()
# (red = behavioural, blue = non-behavioural)

# Plot CDFs of behavioural and non-behavioural input samples:
RSA_tr.RSA_plot_thres(X_P2, idxb_P2, X_Labels=X_Labels)
plt.title("CDFs with new parameterisation, N = %d" % N2 + ', ' + Y_Label + " = %2.2f" % threshold, \
          loc='right')
plt.show()
# (red = behavioural, blue = non-behavioural)

# Compute mean and confidence intervals of the sensitivity indices (mvd,
# maximum vertical distance or KS statistics) across the bootstrap resamples:
mvd_P2_m, mvd_P2_lb, mvd_P2_ub = aggregate_boot(mvd_P2, alfa=alfa) # shape (M,)

# Plot the sensitivity indices with their 95% confidence intervals:
plt.figure()
pf.boxplot1(mvd_P2_m, S_lb=mvd_P2_lb, S_ub=mvd_P2_ub, X_Labels=X_Labels, \
            Y_Label='KS statistic')
plt.title("KS with 95%% CI, new par, N = %d" % N2 + ', ' + Y_Label + " = %2.2f" % threshold)
plt.show()

# TO DO:
# - Is the sample size large enough?
# - How have the sensitivity indices changed by changing the range of variability?

#%% 
#==============================================================================
# 4. Regional Sensitivity Analysis based on grouping
#==============================================================================

# In this section, we assess the sensitivity using the RSA method based on 
# grouping. This variant does not require the definition of a single threshold
# on the output value. It consists of splitting the input factor sample into 
# a given number of groups (e.g. ten) according to the associated output value. 
# In our application, we use ten intervals of increasing output value, 
# designed so to have an equal number of data points in each group. The 
# corresponding ten CDFs are then derived for each input factor.

# We can apply this method to the input-output sample that we have already 
# created.

# Select the output metric to be analysed (uncomment option you want to view):
Y_N2 = YY_N2[:, 0]; Y_Label = 'RMSE' # RMSE
# Y_N2 = YY_N2[:, 1]; Y_Label = 'BIAS' # BIAS

# Set number of groups:
ngroup = 10 
# Perform RSA based on grouping:
_, _, _, _, _, _, idx_gp, Yk = RSA_gp.RSA_indices_groups(X_N2, Y_N2, ngroup=ngroup)
# idx_gp: indices of the ten groups
# Yk: range of values of the output
    
# Look at the range of values of the output in the ten groups:
print(Yk)

# Plot parameter CDFs corresponding to the ten groups:
RSA_gp.RSA_plot_groups(X_N2, idx_gp, Yk, X_Labels=X_Labels, legend_title=Y_Label)
plt.title("CDFs with N = %d" % N2 + ', ' + Y_Label, loc='right')
plt.show()

# TO DO: 
# - From the CDF plots, which parameter would you say is most influential? Why?
# - Are the results of RSA based on grouping consistent with the results of RSA
#   based on threshold?
# - Which additional information is provided by RSA based on grouping?

# Here we only performed a visual analysis of the ten CDFs, but sensitivity 
# indices could also be computed based on the KS statistic.

#%%
#==============================================================================
# 5. Variance-Based Sensitivity Analysis
#==============================================================================

# In this section, we use Variance-Based Sensitivity Analysis to measure output
# sensitivities. We compare with RSA results to assess the effect of the choice
# of GSA method.

# Perform resampling:
# We can re-use the matrix X we have already generated for RSA and split it into
# the two matrices XA and XB. This has the advantage that we have already
# executed the model against these matrices, so that we will only have to 
# execute the model against XC.

# Split X into XA and XB, and obtain the resampling matrix XC:
XA, XB, XC = VB.vbsa_resampling(X_N2)
# Check input sample dimensions:
print(XA.shape)
print(XB.shape)
print(XC.shape)

# Recover samples corresponding to sub-matrices XA and XB:
YYA = YY_N2[0:int(N2/2), :]
YYB = YY_N2[int(N2/2):N2, :]

# Execute the new input samples (XC):
myfun = HyMod.hymod_MulObj # Function used to compute the model output;
# to check what this function does, type:
# help(HyMod.hymod_MulObj)
# Execute the model (WARNING: it may take some time to run this line):
YYC = model_execution(myfun, XC, rain, evap, flow, warmup)

# Select output:
#i_out = 0; Y_Label='RMSE' # RMSE
i_out = 1; Y_Label='RMSE' # BIAS
YA = YYA[:, i_out]
YB = YYB[:, i_out]
YC = YYC[:, i_out]

# Check the model output distribution 
Y_plot = np.concatenate((YA, YC))
plt.figure()
plt.subplot(121)
pf.plot_cdf(Y_plot, Y_Label=Y_Label)
plt.subplot(122)
_, _ = pf.plot_pdf(Y_plot, Y_Label=Y_Label)
plt.show()

# TO DO:
# - Is the distribution multi-modal or highly skewed?
# - Is the variance-based approach adequate?

# Approximate first-order and total-order sensitivity indices using bootstrapping:
Nboot = 1000 # Choose number of bootstrap resamples
# Compute sensitivity indices for Nboot bootstrap resamples:
Si, STi = VB.vbsa_indices(YA, YB, YC, M, Nboot=Nboot)
# Si: first order, shape (Nboot, M)
# STi: first order, shape (Nboot, M)

# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
Si_m, Si_lb, Si_ub = aggregate_boot(Si) # shape (M,)
STi_m, STi_lb, STi_ub = aggregate_boot(STi) # shape (M,)

# Plot bootstrapping results:
plt.figure() # plot main and total separately
plt.subplot(121)
pf.boxplot1(Si_m, S_lb=Si_lb, S_ub=Si_ub, X_Labels=X_Labels, \
            Y_Label='main effects')
plt.title("N = %d" % N2 + ', ' + Y_Label)
plt.ylim([-1, 1.5])
plt.subplot(122)
pf.boxplot1(STi_m, S_lb=STi_lb, S_ub=STi_ub, X_Labels=X_Labels, \
            Y_Label='total effects')
plt.ylim([-1, 1.5])
plt.show()

# TO DO:
# - What is the ranking of input parameters according to these indices, and is
# it consistent with that of RSA?
# - The sensitivity indices should be in the range 0-1. Is this verified here? Why?
# - Are the sensitivity indices robust?

# Repeat VBSA with larger sample size (WARNING: this may take few minutes to run!):
Nnew_vbsa = 10000; # size of new sample
N2_vbsa = Nnew_vbsa + N2 # total sample size

# Add new samples:
Nrep = 5 # set the number of replicate to select the maximin hypercube to
# limit computation time
X_vbsa_2 = AAT_sampling_extend(X_N2, distr_fun, distr_par, N2_vbsa, nrep=Nrep)
# extended sample (it includes the already evaluated samples 'X' and the new ones)

# Extract the new input samples that need to be evaluated:
X_vbsa_new = X_vbsa_2[N2:N2_vbsa, :]

# Resampling strategy:
[XAnew, XBnew, XCnew] = VB.vbsa_resampling(X_vbsa_new)

# Execute the model against the new samples(WARNING: it may take some time to
# run this line)
YYAnew = model_execution(myfun, XAnew, rain, evap, flow, warmup)
YYBnew = model_execution(myfun, XBnew, rain, evap, flow, warmup)
YYCnew = model_execution(myfun, XCnew, rain, evap, flow, warmup)
YAnew = YYAnew[:, i_out]
YBnew = YYBnew[:, i_out]
YCnew = YYCnew[:, i_out]

# Put new and old results together:
YA2 = np.concatenate((YA, YAnew))
YB2 = np.concatenate((YB, YBnew))
YC2 = np.concatenate((np.reshape(YC, (M, int(N2/2))),
                      np.reshape(YCnew, (M, int(Nnew_vbsa/2)))), axis=1)
YC2 = YC2.flatten()

# Recompute indices:
Nboot = 1000
Si2, STi2 = VB.vbsa_indices(YA2, YB2, YC2, M, Nboot)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
Si2_m, Si2_lb, Si2_ub = aggregate_boot(Si2) # shape (M,)
STi2_m, STi2_lb, STi2_ub = aggregate_boot(STi2) # shape (M,)

# Plot sensitivity indices calculated with the extended sample:
plt.figure() # plot main and total effects separately
plt.subplot(121)
pf.boxplot1(Si2_m, S_lb=Si2_lb, S_ub=Si2_ub, X_Labels=X_Labels, \
            Y_Label='main effects')
plt.title("N = %d" % N2_vbsa + ', ' + Y_Label)
plt.ylim([-0.5, 1.2])
plt.subplot(122)
pf.boxplot1(STi2_m, S_lb=STi2_lb, S_ub=STi2_ub, X_Labels=X_Labels, \
            Y_Label='total effects')
plt.ylim([-0.5, 1.2])
plt.show()

# TO DO:
# - Is the sample size adequate now? Does the Variance-Based method require a 
#   smaller or larger sample to reach robust sensitivity indices estimates 
#   compared to RSA?
# - What is the ranking of input parameters according to these indices? 
#   Is it consistent with that of RSA?
# - Are parameters interacting?
# - Now perform the analysis with a different output metric (L664-665).
#   Do the result change depending on the performance metric chosen? Why?