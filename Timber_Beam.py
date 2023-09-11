def Timber_Beam(parameters,Length,Spacing,Cross_section,Smod_factor,Kmod_factor):
    import numpy as np
    L=Length;
    S=Spacing;
    f = parameters[1-1];
    E = parameters[2-1];
    G = parameters[3-1];
    tau = 0.2*f**0.8;
    rho = parameters[4-1];
    q = parameters[5-1];

    B = np.array([47,47,47,47,47,75,75,75,75,75])/1000; # Joist cross-section typical depths (m)
    H = np.array([150,175,200,225,250,150,175,200,225,250])/1000; # Joist cross-section typical width (m)
    
    Cross_section = Cross_section-1;

    A_section = B[Cross_section] * H[Cross_section]; # Area [m2]
    W_section = B[Cross_section] * H[Cross_section]**2 / 6; # Elastic modulus of the section [m3]
    I_section = B[Cross_section] * H[Cross_section]**3 / 12; # Inertia of the section [m4]

    E_fin = E/(1+Smod_factor);
    G_fin = G/(1+Smod_factor);

    if rho<=7 and H[Cross_section]<0.15:
        Kh = min(1.3,(0.15/H[Cross_section])**0.2);
    else:
        Kh=1;

    Kcritic = 1;
    Ksys = 1.1; # to account for different stiffnesses for the different joists
    kcr = 0.67;

    gk1 = self_weight_calculator(H[Cross_section],B[Cross_section],rho);
    gk2 = dead_load_calculator(S,rho);

   
    Load = gk1 + gk2 + q;

    Vslu = Load*L/2; # [kN]
    Shear_stress = 1.5 * Vslu / (kcr * A_section) * 10**-3; # [MPa]
    Shear_stress_capacity = Kmod_factor * Ksys * tau; # [MPa]

    M = Load*L**2/8; # [kNm]
    Bending_stress = M/W_section * 10**-3; # [MPa]
    Bending_stress_capacity = Kh * Kcritic * Kmod_factor * Ksys * f; # [MPa]

    d = 5/384 * Load * L**4 / (E_fin*1000) / I_section + Load * L**2 / 8 / (G_fin*1000) / A_section; # [m]
    Displacement_capacity = 1/200 * L ;

    Safety_factor_sls   = Displacement_capacity / d;
    Y_SLS = Safety_factor_sls; 

    Safety_factor_uls_V = Shear_stress_capacity / Shear_stress;
    Safety_factor_uls_M = Bending_stress_capacity / Bending_stress;
    Safety_factor_uls = min(Safety_factor_uls_V,Safety_factor_uls_M);
    Y_ULS = Safety_factor_uls; 
    
    Y = np.nan * np.ones((2,))
    Y[0]=Y_ULS;
    Y[1]=Y_SLS;
    
    #Y = np.nan * np.ones((1,))'
    #Y[0]=Y_ULS;
    #Y[1]=Y_SLS;
    
    return Y

   
def self_weight_calculator(H,B,rho):
    W = H*B*rho; # [kN/m]
    return W

def dead_load_calculator(S,rho):
    D_Soil = 21 * 0.10;         # Self weight * thickness [kN/m3 * m];
    D_Sand_drainage = 21*0.05;  # Self weight * thickness [kN/m3 * m];
    D_Insulation_and_Membrane = 0.2; # Assumed [kN/m2];
    Deck_tickness_nominal = 0.025;
    Deck_tickness_simulated = Deck_tickness_nominal;
    D_CLT_Deck = rho*Deck_tickness_simulated; #Self weight * thickness [kN/m3 * m];
    D_Services = 0.2; # Assumed [kN/m2];
    D = (D_Soil+D_Sand_drainage+D_Insulation_and_Membrane+D_CLT_Deck+D_Services)*S; #[kN/m]
    return D