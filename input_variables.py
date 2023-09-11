def Normal_to_Lognormal(mu,cov):
    import numpy as np
    beta = np.sqrt(np.log(1+cov**2));
    eta  = np.exp(np.log(mu)-(beta**2)/2);
    return [eta, beta]

def input_variables():
    # Bending strength - fiber parallel
    # index = find(materials.material==timber);
    # fmk = materials.fmkMpa(index); % Acquisition of the characteristic value from the table
    fmean = 25.9; # From Raffaele's plot [MPa]
    COV_fmean = 0.25; # Table 2 JCSS https://www.jcss-lc.org/publications/jcsspmc/timber.pdf
    [eta_f,beta_f]=Normal_to_Lognormal(fmean,COV_fmean);
    
    # Young's modulus
    Emean = 7000; # [MPa]
    COV_Emean = 0.13; # Table 2 JCSS https://www.jcss-lc.org/publications/jcsspmc/timber.pdf
    [eta_Emean,beta_Emean]=Normal_to_Lognormal(Emean,COV_Emean);
    
    # Shear elastic modulus
    Gmean = 440; # [MPa]
    COV_Gmean = 0.13; # Table 1 JCSS https://www.jcss-lc.org/publications/jcsspmc/timber.pdf
    [eta_Gmean,beta_Gmean]=Normal_to_Lognormal(Gmean,COV_Gmean);
    
    # Density
    rho_mean = 3.5; # [kN/m3];
    COV_rho_mean = 0.1; # Table 2 JCSS https://www.jcss-lc.org/publications/jcsspmc/timber.pdf
    sigma_rho = rho_mean*COV_rho_mean;
    
    # Snow Load
    eta_snow = 0.087;  # Median kN/m2 - good for a 98% characteristic value
    beta_snow = 0.6;   # Logarithmic standard deviation
    
    return [eta_f, beta_f, eta_Emean, beta_Emean, eta_Gmean, beta_Gmean, rho_mean, sigma_rho, eta_snow, beta_snow]
