import os
import pandas as pd

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

# Get result Demand/Capacity from .csv file
df = pd.read_csv("solution_file_SAFE.csv")
print(df)
adequacy_factor = df['Answer'].values[0]
print("The Adequacy Factor is: ", adequacy_factor)