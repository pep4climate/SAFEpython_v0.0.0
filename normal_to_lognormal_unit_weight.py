### Import libraries
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

### Switch from a normal to lognormal distribution
## Find parameters of an equivalent Normal distribution
def Normal_to_Lognormal(mu,cov):
    import numpy as np
    # Normal distribution with parameters mu, mean, and cov, coefficient of variation (=standard deviation/mean)
    # Normal distribution with parameters eta and beta, which underlies the Lognormal distribution
    beta = np.sqrt(np.log(1+cov**2)); # also referred as zeta
    eta  = np.log(mu) - 1/2 * (beta**2); # also referred as lambda
    return [eta, beta]

## Input parameters of a pure normal distribution
mean = 22 # mean of the normal distribution
standard_deviation = 0.05 # standard deviation of the normal distribution
coefficient_of_variation = standard_deviation/mean # coefficient of variation of the normal distribution
print("coefficient of variation_normal is: ", coefficient_of_variation)

## Define the value of random variable of interest
maximum_drainage_capacity = 20 # value of interest of the random variable

### Estimate the probability for the interested value of random variable

## Normal distribution
# https://stackoverflow.com/questions/12412895/how-to-calculate-probability-in-a-normal-distribution-given-mean-standard-devi
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
probability_of_flooding_normal = 1 - st.norm(mean, standard_deviation).cdf(x=maximum_drainage_capacity)
print("probability_of_flooding_normal is: ", probability_of_flooding_normal)

### Lognormal distribution
## Switch from a normal to lognormal distribution
beta = np.sqrt(np.log(1+coefficient_of_variation**2)) # standard deviation of a normal distribution equivalent to a lognormal distribution
eta = np.log(mean) - (1/2) * (beta**2) # mean of a normal distribution equivalent to a lognormal distribution
print("beta is: ", beta)
print("eta is: ", eta)

## Calculate paremeters of the lognormal distribution
median_log = np.exp(eta) # median of the lognormal distribution
mean_log = median_log * np.sqrt(1+coefficient_of_variation**2) #mean of the lognormal distribution
standard_deviation_log = np.sqrt(mean_log**2 * np.exp(beta**2)-mean_log**2) #standard deviation of the lognormal distribution
print("median_log is: ", median_log)
print("mean_log is: ", mean_log)
print("standard_deviation_log is: ", standard_deviation_log)

## Calculate natural logarithm of the value of interested of random variable
ln_maximum_drainage_capacity = np.log(maximum_drainage_capacity)

## First way
probability_of_flooding_lognormal = 1 - st.norm(eta, beta).cdf(x=ln_maximum_drainage_capacity)
print("probability_of_flooding_lognormal is: ", probability_of_flooding_lognormal)

## Second way
scale = np.exp(eta)
shape = (beta)
loc = 0

probability_of_flooding_lognormal_second = 1 - st.lognorm(shape, loc, scale).cdf(x=(maximum_drainage_capacity))
print("probability_of_flooding_lognormal_second is: ", probability_of_flooding_lognormal_second)

### Sampling
# https://www.geeksforgeeks.org/how-to-generate-random-numbers-from-a-log-normal-distribution-in-python/
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html
N = 10000 # number of realisations
np.random.seed(1) # seed to reproduce same random numbers each time, it can be any integer number

## Realisations from a normal distribution
s_normal = np.random.normal(loc=mean, scale=standard_deviation, size=N)

# Histogram of realisations
n=30 # number of histogram bins
count, bins, ignored = plt.hist(s_normal, n, density=True, color='green')
x_normal = np.linspace(min(bins), max(bins), N)
pdf_normal = st.norm(mean, standard_deviation).pdf((x_normal))

# Plot probability density function and realisations
# plt.plot(x_normal, pdf_normal, color='black')
# plt.grid()
# plt.show()

## Realisations of a lognormal distribution
# Generate random realisations
s_lognormal = np.random.lognormal(mean=eta, sigma=beta, size=N)

# Histogram of realisations
count, bins, ignored = plt.hist(s_lognormal, n, density=True, color='blue')

# Create the probability density function to double-check realisation

# First way
# np.seterr(invalid='ignore')
# pdf_lognormal = (np.exp((-(np.log(x_normal)-eta)**2)/(2*beta**2))) / (x_normal * beta * np.sqrt(2 * np.pi)) # it leads to a correct lognormal pdf
# pdf_lognormal = st.norm(eta, beta).pdf(np.log(x_normal) # it leads to a wrong lognormal pdf
# pdf_lognormal = (1/x_normal) * st.norm(eta, beta).pdf(np.log(x_normal)) # a lognormal pdf can be obtained by dividing for x a normal pdf

# Plot of probability density function and realisations
# plt.plot(x_normal, pdf_normal, color='black')
# plt.plot(x_normal, pdf_lognormal, color='red')
# plt.grid()
# plt.show()

# Second way
pdf_lognormal_third = st.lognorm(shape, loc, scale).pdf(x_normal)

# Plot of probability density function and realisations
plt.plot(x_normal, pdf_normal, color='black')
plt.plot(x_normal, pdf_lognormal_third, color='magenta')
plt.grid()
plt.show()