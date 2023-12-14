# CSC466FinalProject
Aditi Gajjar, Anagha Sikha, Othilia Norell, Soren Paetau, Nicholas Tan \\ agajjar@calpoly.edu / arsikha@calpoly.edu \ onorell@calpoly.edu  / spaetau@calpoly.edu / nktan@calpoly.edu
​

## HOW TO RUN CODE

### Q1: (Random Forest on Feature Importance)
- run python3 randomForest.py <numAttributes> <numDatapoints> <numTrees>
To run code on different subsets of attributes (all, without best attributes, and without worst attributes) by updating line 34 in randomForest.py – instructions in randomForest.py

### Q2 (Apriori Rules): All code is contained within a jupyter notebook

### Q3 (Clustering): 
- For KMeans, run python3 kmeans.py spotify_songs.csv <k> [-p (kmeans_plus)] [-m (manhattan dist)] [-n (normalize data)] [-t (testing)]
- For DBScan, run python3 dbscan.py spotify_songs.csv <epsilon> <NumPoints> [-m (manhattan dist)]
    
### Q4 (Decision Tree vs Random Forest): 
- To run C4.5 run python3 InduceC45.py spotify_new_train.csv in the terminal
- To run classify on the decision tree from InduceC45.py run on the test dataset: python3 classify.py spotify_new_test.csv spotify_new_train.json
- To run Random Forests run python3 randomForest.py spotify_songs_new.csv <numAttributes> <numDataPoints> <numTrees> (example: python3 randomForest.py spotify_songs_new.csv 9 600 50)
- Results of the random forests classification are found in spotify_songs_new_rf_all.txt for all 9 attributes
- From hypertuning results of the random forests classification are found in spotify_songs_new_rf.txt for 4 attributes

### Q5 (Collaborative Filtering): 
-Reference readme in CollabFiltering
-use -sp with testcases given


