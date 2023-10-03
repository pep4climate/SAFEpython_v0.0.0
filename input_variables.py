def Normal_to_Lognormal(mu,cov):
    import numpy as np
    # Normal distribution with parameters mu, mean, and cov, coefficient of variation (=standard deviation/mean)
    # Lognormal distribution with parameters eta and beta
    beta = np.sqrt(np.log(1+cov**2)); # also referred as zeta
    eta  = np.log(mu) - 1/2 * (beta**2); # also referred as lambda
    scale = np.exp(eta)
    shape = (beta)
    loc = 0
    # return [shape, scale]
    return [scale, beta]

def input_variables():
    # Bending strength - fiber parallel
    # index = find(materials.material==timber);
    # fmk = materials.fmkMpa(index); % Acquisition of the characteristic value from the table
    # fmean = 25.9; # From Raffaele's plot [MPa]
    # COV_fmean = 0.25; # Table 2 JCSS https://www.jcss-lc.org/publications/jcsspmc/timber.pdf
    # [eta_f,beta_f]=Normal_to_Lognormal(fmean,COV_fmean);
    
    # Masonry unit weight
    Wmean = 22; # [kN/m3]
    COV_Wmean = 0.05; # Table 2.1.1. JCSS PART II - LOAD MODELS - 2.01
    # [shape, scale]=Normal_to_Lognormal(Wmean,COV_Wmean)
    [median_Wmean,beta_Wmean]=Normal_to_Lognormal(Wmean,COV_Wmean)
    
    # Angle of friction
    Phimean = 38.66; # [deg degrees]
    COV_Phimean = 0.19; # Table 3.2-13 JCSS PART III - RESISTANCE MODELS - 3.2 MASONRY PROPERTIES
    [median_Phimean, beta_Phimean]=Normal_to_Lognormal(Phimean,COV_Phimean)
    
    # Young's modulus
    # Emean = 7000; # [MPa]
    # COV_Emean = 0.13; # Table 2 JCSS https://www.jcss-lc.org/publications/jcsspmc/timber.pdf
    # [eta_Emean,beta_Emean]=Normal_to_Lognormal(Emean,COV_Emean);
    
    # Shear elastic modulus
    # Gmean = 440; # [MPa]
    # COV_Gmean = 0.13; # Table 1 JCSS https://www.jcss-lc.org/publications/jcsspmc/timber.pdf
    # [eta_Gmean,beta_Gmean]=Normal_to_Lognormal(Gmean,COV_Gmean);
    
    # Density
    # rho_mean = 3.5; # [kN/m3];
    # COV_rho_mean = 0.1; # Table 2 JCSS https://www.jcss-lc.org/publications/jcsspmc/timber.pdf
    # sigma_rho = rho_mean*COV_rho_mean;
    
    # Snow Load
    # eta_snow = 0.087;  # Median kN/m2 - good for a 98% characteristic value
    # beta_snow = 0.6;   # Logarithmic standard deviation
    
    # return [shape, scale]
    return [median_Wmean, beta_Wmean, median_Phimean, beta_Phimean]
    # return [eta_f, beta_f, eta_Emean, beta_Emean, eta_Gmean, beta_Gmean, rho_mean, sigma_rho, eta_snow, beta_snow]
