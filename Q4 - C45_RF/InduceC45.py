import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import numpy as np
import math
import json
import sys

class DecisionTreeClassifier():
    def entropy(self, D):
        class_labels_counts = D.iloc[:,-1].value_counts().to_dict()

        entropy = 0
        for val in class_labels_counts.values():
            prob = val/len(D)
            entropy += prob * math.log2(prob)
        return -entropy
    
    def findBestSplit(self, D, a): # finds best binary split for a continuous attribute
        p0 = self.entropy(D)
        gains = dict.fromkeys(sorted(list(D[a].unique())), 0)

        for val in gains.keys():
            left = D[D[a] <= val]
            right = D[D[a] > val]
            gains[val] = p0 - len(left)/len(D) * self.entropy(left) - len(right)/len(D) * self.entropy(right)
        max_value = max(gains, key=gains.get)
        max_gain = gains[max_value]
        return max_value, max_gain
    
    def selectSplittingAttribute(self, D, A, threshold): # uses information gain
        p0 = self.entropy(D)
        entropies = dict.fromkeys(A, 0)
        gains = dict.fromkeys(A, 0)

        for a in A:
            if is_numeric_dtype(D[a]): # if continuous
                gains[a] = self.findBestSplit(D, a)[1]
            else:
                split = dict.fromkeys(D[a].unique(), [])
                for d in split.keys():
                    part = D[D[a] == d]
                    split[d] = [len(part)/len(D), self.entropy(part)]
                for val in split.values():
                    entropies[a] += val[0] * val[1]
                gains[a] = p0 - entropies[a]
        best = max(gains, key=gains.get)
        # print(gains)
        if gains[best] > threshold:
            # print("in best", gains[best], threshold)
            return best
        else:
            # print("not")
            return None
        
    def C45(self, D, A, og_D, threshold, parent_val=None, parent_var=None, sign=None):
        T = {}
        class_labels_counts = D.iloc[:,-1].value_counts().to_dict()
        c = max(class_labels_counts, key=class_labels_counts.get)

        # Step 1: check termination conditions
        if len(class_labels_counts.keys()) == 1:
            # print("case 1", class_labels_counts.keys())
            T['decision'] = list(class_labels_counts.keys())[0]
            if is_string_dtype(og_D[parent_var]):
                T['p'] = og_D[og_D[parent_var] == parent_val].iloc[:,-1].value_counts()[T['decision']]/og_D[parent_var].value_counts()[parent_val]
            elif sign == '<=':
                T['p'] = og_D[og_D[parent_var] <= parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] <= parent_val])
            else:
                T['p'] = og_D[og_D[parent_var] > parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] > parent_val])

        elif A == []:
            print("case 2")
            T['decision'] = c
            if is_string_dtype(og_D[parent_var]):
                T['p'] = og_D[og_D[parent_var] == parent_val].iloc[:,-1].value_counts()[T['decision']]/og_D[parent_var].value_counts()[parent_val]
            elif sign == '<=':
                T['p'] = og_D[og_D[parent_var] <= parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] <= parent_val])
            else:
                T['p'] = og_D[og_D[parent_var] > parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] > parent_val])
        # Step 2: select splitting attribute
        else: 
            # print("case3")
            best = self.selectSplittingAttribute(D, A, threshold)
            if best is None: # no attribute is good for a split
                if (parent_val is None) and (parent_var is None):
                    print('Threshold value too high to even select a root node')
                    exit()
                T['decision'] = c
                if is_string_dtype(og_D[parent_var]):
                    T['p'] = og_D[og_D[parent_var] == parent_val].iloc[:,-1].value_counts()[T['decision']]/og_D[parent_var].value_counts()[parent_val]
                elif sign == '<=':
                    T['p'] = og_D[og_D[parent_var] <= parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] <= parent_val])
                else:
                    T['p'] = og_D[og_D[parent_var] > parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] > parent_val])
            # Step 3: tree construction
            else:
                T['var'] = best
                T['edges'] = []
                if is_string_dtype(D[best]): # categorical
                    for i, v in enumerate(D[best].unique()):
                        D_v = D[D[best] == v]
                        T['edges'].append({})
                        T['edges'][i]['edge'] = {}
                        T['edges'][i]['edge']['value'] = v
                        if not D_v.empty:
                            copy = A.copy()
                            copy.remove(best)
                            T_v = self.C45(D_v, copy, og_D, threshold, v, best) # recursive call
                            if 'decision' in T_v.keys():
                                T['edges'][i]['edge']['leaf'] = T_v
                            else:
                                T['edges'][i]['edge']['node'] = T_v
                        else: # ghost leaf node
                            T['edges'][i]['edge']['leaf']['decision'] = c
                            T['edges'][i]['edge']['leaf']['p'] = og_D[og_D[parent_var] == parent_val].iloc[:,-1].value_counts()[T['decision']]/og_D[parent_var].value_counts()[parent_val]
                else: # continuous
                    v = self.findBestSplit(D, best)[0]
                    for i in range(2):
                        T['edges'].append({})
                        T['edges'][i]['edge'] = {}
                        if i == 0:
                            D_v = D[D[best] <= v]
                            sign = '<='
                        else:
                            D_v = D[D[best] > v]
                            sign = '>'
                        T['edges'][i]['edge']['value'] = sign + ' ' + str(v)
                        if not D_v.empty:
                            T_v = self.C45(D_v, list(D_v.columns)[:-1], og_D, threshold, v, best, sign) # recursive call
                            if 'decision' in T_v.keys():
                                T['edges'][i]['edge']['leaf'] = T_v
                            else:
                                T['edges'][i]['edge']['node'] = T_v
                        else: # ghost leaf node
                            T['edges'][i]['edge']['leaf']['decision'] = c
                            if sign == '<=':
                                T['edges'][i]['edge']['leaf']['p'] = og_D[og_D[parent_var] <= parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] <= parent_val])
                            else:
                                T['edges'][i]['edge']['leaf']['p'] = og_D[og_D[parent_var] > parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] > parent_val])
        return T

if __name__ == '__main__':
    # from command line
    if (len(sys.argv) < 2) or (len(sys.argv) > 3):
        print('Wrong format')
        sys.exit(1)

    csv_file = sys.argv[1]
    before = csv_file.split('.csv')[0]
    dataset = before.split('/')[-1]

    if len(sys.argv) == 3:
        restrictions_file = sys.argv[2]
    else:
        restrictions_file = None

    # getting values
    D = pd.read_csv(csv_file)
    og_D = D.copy()
    if restrictions_file is None: # assumes all
        #A = list(D.columns)
        A = list(D.columns[:-1])
    else:
        restrict = open(restrictions_file, 'r').read().split(',')
        restrict = [int(r) for r in restrict]

        A = []
        for index, a in enumerate(D.columns):
            if restrict[index] == 1:
                A.append(a)

    # run C4.5
    classifier = DecisionTreeClassifier()
    T = classifier.C45(D, A, og_D, threshold=0.01) # spotify
    T = {'dataset': csv_file, 'node': T}
    with open(dataset + '.json', 'w') as outfile:
        json.dump(T, outfile, indent=4)

# python3 InduceC45.py spotify_new_train.csv 
