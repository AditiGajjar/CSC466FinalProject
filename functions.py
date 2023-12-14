#CSC 466 Fall 2023 - Lab 5: Collaborative Filtering
#Soren Paetau, Othilia Norell and Nicholas Tan \\  spaetau@calpoly.edu / onorell@calpoly.edu / nktan@calpoly.edu


"""
Functions designated to be helper functions to other programs
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import random
import csv
import os
from datetime import datetime
from io import StringIO
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, mean_absolute_error, accuracy_score


"""
Nice helper function to clean up mains, just intializes variables
"""
def initilaize_input(arg):
    prin = True #printing each observation
    out_path = None #none should work, since it gets assigned only when writing to csv
    write = False #writes to csv in directory under outputs of currtime
    display_plot = False #display SNS heatmap of confusion matrix, will still output string
    spotify = True
    if "-s" in arg:
        prin = False #silent mode
    if "-w" in arg:
        write = True
        out_path = init_dir() #init_dir will make a directory under outputs of current time, returns that path
    if "-d" in arg:
        display_plot = True
    if "-sp" in arg:
        spotify = True
    
    return prin, out_path, write, display_plot, spotify


"""
Want to return list [(userId, jokeId),...] of length n. Note that userid is a row and joke id is a column, such that the value of D(userId, jokeId) \neq NA. 
"""
def test_sample(df, n):
    cases = []
    numrows = df.shape[0]
    numcols = df.shape[1]
    while len(cases) < n:
        row = random.randint(0, numrows-1) #rows are 0 indexed
        col = random.randint(1, numcols) #columns are 1 indexed
        if not pd.isna(df.at[row,col]) and not (row,col) in cases: #make sure we don't get duplicate observations, unlikely but still important to check
            cases.append((row,col))

    return cases

"""
Want to return index of N nearest neighbors, along with similarity values to C of that matr
"""

def print_methods():
    print("Collaborative Filtering Methods:")
    print("1. Mean Utility: Average of all other users response for item")
    print("2. Weighted Sum: Implements Pearson similarity as weighting of other responses")
    print("3. Adjusted Weighted Sum: Above but takes average user reponse into account ") 
    print("4. Weighted N-nn: Above but only looks at 10 most similar responses")
    print("5. Adjusted Weight N-nn Sum: Same as above") 


"""
writes output csv to path
"""
def write_output(df, path, iter):
    name = f"iter{iter}.csv"
    new_path = os.path.join(path, name)
    df.to_csv(new_path, index=False)
    print(f"df written to {new_path}")



def write_mets(metrics, out_path): #metrics is of form [(precision, recall, f1, accuracy, mae),.....)
    cols = ["precision", "recall", "f1", "accuracy", "mae"]
    df = pd.DataFrame(metrics, columns=cols)
    name = "metrics.csv"
    new_path = os.path.join(out_path, name)
    df.to_csv(new_path)

    
"""
iniitalizers the time and date custom folder, important in case of running multiple in CF random, will store are iter's in same directory.
"""
def init_dir():
    curr_time = datetime.now()
    folder_name = curr_time.strftime("%d_%H-%M-%S")
    outputs_folder = os.path.join(os.getcwd(), "outputs")

    folder_path = os.path.join(outputs_folder, folder_name)
    os.mkdir(folder_path)
    print(f"Successfully created folder: {folder_name}")
    return folder_path


"""
parses in test files, dropping any if NA
"""
def parse_tests(filename, df):
    cases = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)

        for line in csv_reader:
            l1 = int(float(line[0]))
            l2 = int(float(line[1]))
            if not pd.isna(df.at[l1, l2]): #cleans to make sure test cases are valid, just skips ones that arent
                cases.append((l1, l2))
    return cases

"""
Returns dataframe, rows index 0-n and cols indexed 1-100, with na values replaced for 99!
"""
def parse_data(path):
    df = pd.read_csv(path, header=None)
    df.replace(99, pd.NA, inplace=True)
    num_rev = df.pop(0) #first column is number of reviews, not needed
    return df


def parse_spot(spot):
    df = pd.read_csv(spot, header=None, skiprows=1)
    return df
