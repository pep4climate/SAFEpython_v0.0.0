# import os

 

# def bridge_assessment(parameters):

#     sw = parameters[0];

#     # ricordati che sw deve essere concatenato con il 20 nel seguito

#     os.system("geo64 –p:uW:866=20 –x –sf:”_mod_uW” rotations_arch.geo")

#     # dopo aver girato devi aprire il file di output e acquisire Y

#     return Y

def bridge_assessment():

    # https://docs.python.org/3/tutorial/stdlib.html
    import os

    # To read csv files
    import pandas as pd

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

    uW = 24 # Attempt to define uW as a variable outside of os.system(<_>)

    # Launch LimitState:GEO
    # Get a list of files within the ...\bin folder
    # The modified variable is uw, that is unit weight of masonry of tunnel in file Tunnel.geo
    # os.system("geo32.exe -p:uw:2594=24 -x -sf:_my_test_SAFE -sl:solution_file_SAFE.csv Tunnel.geo") # any new instance does not overwrite the .csv file, but append result to it as a new row
    # os.system("geo32.exe -p:'uw:2594=24' -x -sf:'my_test_SAFE -sl:solution_file_SAFE.csv Tunnel.geo") # it works with/without '' around uw
    # https://stackoverflow.com/questions/71403394/how-do-i-put-a-variable-in-a-path
    # GEO support team said: sometimes quotes are needed, other not
    os.system("geo32.exe -p:'uw:2594={uW}' -x -sf:_my_test_SAFE -sl:solution_file_SAFE.csv Tunnel.geo")

    # Get result Demand/Capacity from .csv file
    df = pd.read_csv("solution_file_SAFE.csv")
    print(df)
    adequacy_factor = df['Answer'].values[0]
    print("The Adequacy Factor is: ", adequacy_factor)
    
    Y = adequacy_factor

    return Y

# Call the function bridge_assessment() when it is launched from this file
bridge_assessment()