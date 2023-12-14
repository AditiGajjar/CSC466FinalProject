#CSC 466 Fall 2023 - Lab 5: Collaborative Filtering
#Soren Paetau, Othilia Norell and Nicholas Tan \\  spaetau@calpoly.edu / onorell@calpoly.edu / nktan@calpoly.edu

#HOW TO RUN: 
#python3 EvaluateCFList Method Filename

from evaluates import *

def main():
    if len(sys.argv) == 1:
        print_methods()
        print("EvaluateCFList Method Filename [-w (write_to_csv)] [-s (silent - only shows accuracy metrics)] [-d (display cm plot)]")
        quit()

    prin, out_path, write, display_plot, spot = initilaize_input(sys.argv)

    path = "jester-data-1.csv"
    df = parse_data(path)
    if spot:
        path = "spotify_ratings.csv"
        df = parse_spot(path)
    method = int(sys.argv[1])
    test_path = sys.argv[2]
    tests = parse_tests(test_path, df) #should be cleaned
    
    avrgs = calc_avrgs(df, method)
    preds = evaluate(method, df, tests, avrgs)

    evaluate_output(df, tests, preds, out_path, prin, write, display_plot)

if __name__ == "__main__":
    main()
