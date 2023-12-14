import pandas as pd
import sys
import numpy as np
import random
from InduceC45 import DecisionTreeClassifier
from classify import classify

class RFClassifier():
    def __init__(self, m, k, N):
        self.m = m
        self.k = k
        self.N = N
       
    def data_selection(self, training):
        cols = random.sample(list(training.columns)[:-1], self.m) # without replacement
        cols.append(list(training.columns)[-1])
        rows = list(np.random.choice(range(len(training)), size=self.k, replace=True)) # with replacement
        return training.loc[rows, cols]

    def random_forest(self, training):
        forest = [] # list that contains all the trees
        for i in range(self.N):
            tree_data = self.data_selection(training)
            og_tree_data = tree_data.copy()
            classifier = DecisionTreeClassifier()
            print("Building tree {}/{}".format(i + 1, self.N))
            # tree = classifier.C45(tree_data, list(tree_data.columns)[:-1], og_tree_data, threshold=0.1) # iris
            #tree = classifier.C45(tree_data, list(tree_data.columns)[:-1], og_tree_data, threshold=0.05) # heart
            tree = classifier.C45(tree_data, list(tree_data.columns)[:-1], og_tree_data, threshold=0.04) # credit
            forest.append(tree)
        return forest

    def cross_validation(self):
        indices = list(D.index.values)[1:]
        random.shuffle(indices)
        n = 10
        folds = [indices[i::n] for i in range(n)]

        correct_counts = []
        accuracies = []
        cf_matrices = []

        file_name = dataset + '_rf_all.txt'
        with open(file_name, 'w') as file:
            f = 1
            for fold in folds:
                print(f"Processing fold {f}/{len(folds)}")

                file.write('Fold' + ' ' + str(f) + ':' + '\n')
                training = D.drop(fold, axis=0).reset_index(drop=True)
                holdout = D.iloc[fold].reset_index(drop=True)
                forest = self.random_forest(training)

                y_pred = []
                y = holdout.iloc[:,-1]
                correct = 0
                incorrect = 0
                for i in range(len(holdout)):
                    file.write(' '.join(map(str, holdout.iloc[i, :-1])) + ' ')
                    actual = y[i]
                    tree_preds = []
                    for tree in forest:
                        pred = classify(holdout.iloc[i, :-1].to_dict(), tree)
                        tree_preds.append(pred)
                    most_common_pred = max(set(tree_preds), key=tree_preds.count)
                    file.write(str(most_common_pred) + '\n')
                    if actual == most_common_pred:
                        correct += 1
                    else:
                        incorrect +=1
                    y_pred.append(most_common_pred)

                correct_counts.append(correct)
                accuracies.append(correct/len(holdout))
                cf_matrices.append(pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted']))
                file.write('\n')
                f += 1
                print(f"Fold {f}/{len(folds)} completed.")
            
            cf_matrix = sum(cf_matrices)

            file.write('\n' + 'Overall confusion matrix: ' + '\n')
            file.write(str(cf_matrix) + '\n')
            i = 0
            for label in cf_matrix.index:
                TP = cf_matrix.iloc[i, i]
                FP = np.sum(cf_matrix.iloc[:, i]) - TP 
                FN = np.sum(cf_matrix.iloc[i, :]) - TP
                TN = cf_matrix.values.sum() - TP - FP - FN
                file.write('\n' + "For class label '" + str(label) + "': " + '\n')
                file.write('Confusion matrix: ' + '\n')
                temp = [[TP, FP], 
                        [FN, TN]]
                i += 1
                for row in temp:
                    file.write(str(row) + '\n')
            file.write('\n' + 'Overall accuracy: ' + str(sum(correct_counts)/len(D)) + '\n')
            file.write('Average accuracy: ' + str(sum(accuracies)/n) + '\n')

if __name__ == '__main__':
    # from command line, has four inputs
    if len(sys.argv) != 5:
        print('Wrong format')
        sys.exit(1)
    
    # getting values
    csv_file = sys.argv[1]
    before = csv_file.split('.csv')[0]
    dataset = before.split('/')[-1]

    m = int(sys.argv[2])
    k = int(sys.argv[3])
    N = int(sys.argv[4])

    D = pd.read_csv(csv_file)
    
    # check that input parameters are valid
    if m > len(list(D.columns)[:-1]):
        print('m is larger than number of attributes, please provide a smaller m')
    if k > len(D):
        print('k is larger than number of data points, please provide a smaller k')

    # run cross_validation
    classifier = RFClassifier(m, k, N)
    classifier.cross_validation()

# python3 randomForest.py spotify_songs_new.csv 9 600 50