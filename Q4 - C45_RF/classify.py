import pandas as pd
import json
import sys

def classify(data, T):
    # reached a leaf node, return the classification result
    if 'decision' in T:
        return T['decision']
    
    if 'var' in T:
        value = data.get(T['var'])
    
    if 'edges' in T:
        for dic in T['edges']:
            if dic['edge']['value'] == value:
                return classify(data, dic)
            elif dic['edge']['value'].split()[0] == '<=' and value <= float(dic['edge']['value'].split()[1]):
                return classify(data, dic)
            elif dic['edge']['value'].split()[0] == '>' and value > float(dic['edge']['value'].split()[1]):
                return classify(data, dic)
                
    if ('edge' in T) and ('leaf' in T['edge']):
        return classify(data, T['edge']['leaf'])
    
    if ('edge' in T) and ('value' in T['edge']):
        return classify(data, T['edge']['node'])
            
if __name__ == '__main__':
    # from command line
    if len(sys.argv) != 3:
        print('Wrong format')
        sys.exit(1)

    csv_file = sys.argv[1]
    json_file = sys.argv[2]

    training = pd.read_csv(csv_file)
    with open(json_file) as file:
        T = json.load(file)

    choice = input("Enter 'normal' or 'silent' for run option: ")

    # if input CSV file is a training set
    y_pred = []
    y = training.iloc[:,-1] 
    correct = 0
    incorrect = 0
    for i in range(len(training)):
        actual = y[i]
        data = training.iloc[i, :-1].to_dict()
        pred = classify(data, T['node'])
        if actual == pred:
            correct += 1
        else:
            incorrect +=1
        y_pred.append(pred)
        
        if choice.lower() == 'normal':
            print('Actual: ')
            print(training.iloc[i])
            print('Predicted: ' + str(pred))
        elif choice.lower() == 'silent':
            print(i, actual, pred)

    print('Total number of records classified: ' + str(len(training)))
    print('Total number of records correctly classified: ' + str(correct))
    print('Total number of records incorrectly classified: ' + str(incorrect))
    print('Overall accuracy of the classifier: ' + str(correct/len(training)))
    print('Overall error rate of the classifier: ' + str(incorrect/len(training)))
    cf_matrix = pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print(cf_matrix)

    # if input CSV file does not contain category attribute
    # y = training 
    # y_pred = []
    # for i in range(len(training)):
    #     actual = y[i]
    #     data = training.iloc[i].to_dict()
    #     pred = classify(data, T['node'])
    #     y_pred.append(pred)

    #     if choice.lower() == 'normal':
    #         print('Predicted: ' + str(pred))
    #     elif choice.lower() == 'silent':
    #         print(i, pred)

# python3 classify.py spotify_new_test.csv spotify_new_train.json