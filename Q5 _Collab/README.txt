#CSC 466 Fall 2023 - Lab 5: Collaborative Filtering
#Soren Paetau, Othilia Norell and Nicholas Tan \\  spaetau@calpoly.edu / onorell@calpoly.edu / nktan@calpoly.edu
​
***HOW TO RUN CODE***
If you want to use -w (write csv's), program requires a directory in the current working directory called 'outputs'. Will then write to a

EvaluateCFRandom.py <method (int)> <size (int)> <repeats (int)> [-w (write_to_csv)] [-s (silent - only shows accuracy metrics)] [-d (display cm plot)]
Evaluates <size> randomly selected points using the given method, repeats process <repeats> times.

EvaluateCFList <method (int)> <filename.csv> [-w (write_to_csv)] [-s (silent - only shows accuracy metrics)] [-d (display cm plot)] [-sp (for final project spotify runs)]

metrics.py 
has hard coded lists of MAE values from all five implemented methods and calculates the standard deviation and mean across the MAE values for each method. 


Methods:
1. Mean Utility

2. Weighted Sum

3. Adjusted Weighted Sum

4. Weighted N-nn

5. Adjusted Weight N-nn sum

*note* N is hardcoded to 10
