# import os

# def bridge_assessment(parameters):

#     sw = parameters[0];

#     # ricordati che sw deve essere concatenato con il 20 nel seguito

#     os.system("geo64 –p:uW:866=20 –x –sf:”_mod_uW” rotations_arch.geo")

#     # dopo aver girato devi aprire il file di output e acquisire Y

#     return Y

def bridge_assessment(realisations):

    # https://docs.python.org/3/tutorial/stdlib.html
    import os

    # To read csv files
    import pandas as pd
    
    import numpy as np

    # https://www.geeksforgeeks.org/clear-screen-python/
    os.system("cls")

    # https://www.tutorialspoint.com/How-to-know-change-current-directory-in-Python-shell#:~:text=Use%20the%20chdir()%20function,absolute%20or%20relative%20path%20argument.
    cwd = os.getcwd()
    print("the cwd is:", cwd)

    # Set LimitState:GEO\bin folder as current working folder
    # geo_dir = "C:\Program Files\LimitState\GEO3.6\bin"
    new_cwd = os.chdir(r"C:\Program Files\LimitState\GEO3.6\bin")
    cwd = os.getcwd()
    print("the new cwd is:", cwd)

    # list of files in current directory
    # https://pynative.com/python-list-files-in-a-directory/#h-get-a-list-of-files-in-current-directory-in-python
    files = []
    for file_path in os.listdir('.'):
        if os.path.isfile(os.path.join('.', file_path)):
            files.append(file_path)

    for file in files:
        print(file)

    # uW = realisations[1-1,] # Attempt to define uW as a variable outside of os.system(<_>)
    # phidash = realisations[2-1] # Attempt to define phi' as a second variable outside of os.sytem(<_>)
    
    N = len(realisations) # number of realisations
    M = int(realisations.size/len(realisations)) # number of random variables
    
    # Launch LimitState:GEO
    # Get a list of files within the ...\bin folder
    # The modified variable is uw, that is unit weight of masonry of tunnel in file Tunnel.geo
    # os.system("geo32.exe -p:uw:2594=24 -x -sf:_my_test_SAFE -sl:solution_file_SAFE.csv Tunnel.geo") # any new instance does not overwrite the .csv file, but append result to it as a new row
    # os.system("geo32.exe -p:'uw:2594=24' -x -sf:'my_test_SAFE -sl:solution_file_SAFE.csv Tunnel.geo") # it works with/without '' around uw
    # https://stackoverflow.com/questions/71403394/how-do-i-put-a-variable-in-a-path
    # GEO support team said: sometimes quotes are needed, other not
    # os.system("geo32.exe -p:'uw:2594={uW}' -x -sf:_my_test_SAFE -sl:solution_file_SAFE.csv Tunnel.geo")
    
    for row in realisations:
        uw = row[0]
        phidash = row[1]
        # sw = parameters[0] # command for one instance of sw
        # phidash = parameters[1] # command for one instance of phidash
        # print(sw)
        # print(phidash)
        os.system("geo32.exe -p:uw:2594=" + str(uw) + " -p:phidash:2596=" + str(phidash) + " -x -sf:_my_test_SAFE_5 -sl:solution_file_SAFE_5.csv Tunnel.geo")
        # string = "geo32.exe -p:uw:2594=" + '{sw}' + " -x -sf:_my_test_SAFE -sl:solution_file_SAFE.csv Tunnel.geo";
        # os.system("geo32.exe -p:uw:2594=" + str(sw) + " -x -sf:_my_test_SAFE -sl:solution_file_SAFE.csv Tunnel.geo") # working instance
        # os.system("geo32.exe -p:'uw:2594={sw}' -x -sf:_my_test_SAFE -sl:solution_file_SAFE.csv Tunnel.geo")
        # exit(0)
# initialise Y as a list/vector
# for loop over the length of X
# and then append results to Y
# load Y in the workflow

    # Get result Demand/Capacity from .csv file
    csv_file = "solution_file_SAFE_5.csv"
    df = pd.read_csv(csv_file)
    column_name = 'Answer'
    adequacy_factor = df[column_name]
    print("The Adequacy Factor is: ", adequacy_factor)
    
    Y = adequacy_factor.to_numpy()
    print("Y is:", Y)

    return Y

# Commands for calling the function bridge_assessment() when it is launched from this file
# uw = 18 # value lower than the one from the file Tunnel.geo
# phidash = 35 # value from the file Tunnel.geo
# parameters = [uw, phidash]
# bridge_assessment(parameters)