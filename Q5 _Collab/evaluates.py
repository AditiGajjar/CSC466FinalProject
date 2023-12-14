#CSC 466 Fall 2023 - Lab 5: Collaborative Filtering
#Soren Paetau, Othilia Norell and Nicholas Tan \\  spaetau@calpoly.edu / onorell@calpoly.edu / nktan@calpoly.edu

from functions import *

def evaluate(method, df, cases, averages):
    #hardcode N!
    n = 10
    if method == 1: #mean utility
        return mean_utility(df, cases)
    elif method == 2: #Weighted Sum
        return weighted_sum(df, cases, averages)
    elif method == 3: #Adjusted Weighted Sum
        return weighted_sum(df, cases, averages, adj = True)
    elif method == 4: #Weighted N-nn
        return weighted_sum(df, cases, averages, N = n, avrgs = averages)
    elif method == 5: #Adjusted Weight N-nn sum
        return weighted_sum(df, cases, averages, N = n, adj = True)


def evaluate_output(df, tests, preds, test_path = "", print_t = True, write_t = False, display_plot = False, iter = ""):
    users = [tup[0] for tup in tests] #list of all user ids
    items = [tup[1] for tup in tests] #list of all joke ids
    actual = [df.at[user, items] for user, items in zip(users, items)] #list of u(c,s), aka rating of joke for each test case -- will never be NAN since tests is cleaned to always have defined

    delta = [pred - actu for pred, actu in zip(preds, actual)] #make absolute?
    res_df = pd.DataFrame({"userID":users, "itemID":items, "Actual_Rating":actual, "Predicted_Rating":preds, "Delta_Rating": delta})
    
    if write_t: #writes output to defined directory, iter is used to label for multiple runs run_i.csv
        write_output(res_df, test_path, iter)

    #fancy looking function that just prints each line
    if print_t:
        csv_output = StringIO()
        res_df.to_csv(csv_output, index=False, header=True)
        csv_lines = csv_output.getvalue().split('\n')
        for line in csv_lines:
            print(line)
    
    #threshold hardcoded by lab results
    threshold = 3

    #gives metrics and such for performance of prediction
    precision, recall, f1, accuracy, mae = evaluate_results(preds, actual, threshold, display_plot, iter)
    
    #unused
    return res_df, (precision, recall, f1, accuracy, mae)


def scores(actual, predicted):
    TP = sum(act == True and pred == True for act, pred in zip(actual, predicted))
    FP = sum(act == False and pred == True for act, pred in zip(actual, predicted))
    TN = sum(act == False and pred == False for act, pred in zip(actual, predicted))
    FN = sum(act == True and pred == False for act, pred in zip(actual, predicted))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    return precision, recall, f1, accuracy

def mean_abs_error(actual, preds):
    #sum of i=1..n |y-x|/n
    n = len(actual)
    total_error = sum(abs(y - x) for y, x in zip(actual, preds))
    mae = total_error / n
    return mae


def evaluate_results(preds, actual, thresh = 5, d = False, iter = ""):
    
    #Boolean values of whether predicted/actual values are above threshold, if true \implies reccomend joke
    print(preds)
    bool_pred = [value >= thresh for value in preds]
    bool_actu = [value >= thresh for value in actual]
    
    #Calculating metrics on our own: 
    mae = mean_abs_error(actual, preds)
    precision, recall, f1, accuracy = scores(bool_actu, bool_pred)


    #USES SKLEARN.METRICS
    cm = confusion_matrix(bool_actu, bool_pred) #we could use scikitlearn to make the confusion matrix
    #mae = mean_absolute_error(actual, preds) #uses scikitlearn
    #precision = precision_score(bool_actu, bool_pred, zero_division = 0.0) 
    #recall = recall_score(bool_actu, bool_pred, zero_division = 0.0)
    #f1 = f1_score(bool_actu, bool_pred, zero_division = 0.0)
    #accuracy = accuracy_score(bool_actu, bool_pred)

    #counts number of correct observations

    print("-------------------------------------------------------------")
    print(f"MAE: {mae:.4}")
    print(f"Reccomending a joke == True with threshold: {thresh}")
    print("Confusion Matrix:")
    print(cm)

    #sneaky plot, unsure how it will work with multiple iterations!
    if d:
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Predicted 0", "Predicted 1"],
                    yticklabels=["Actual 0", "Actual 1"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix {iter}")
        plt.show()


    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\n")

    #unused but could be helpful
    return precision, recall, f1, accuracy, mae


#writes output to the created directory, under the 



"""
Layman terms: returns the average rating for that joke, df has na values for 99, so skipna automatically skips over and averages out
"""
def mean_utility(df, cases):
    results = []
    for case in cases: #case = (userId, colId) both numbers
        joke = case[1]
        res = df[joke].mean(skipna = True)
        results.append(res)
    return results


def calc_avrgs(df, method):
    #concept here is that averages at 0 give same result as if not adjusted
    if method in (3,5): #adjusted methods
        avrgs = df.mean(axis = 1, skipna=True)
    else:
        avrgs = np.zeros(df.shape[0]) #redundant and big memory usage but should be ok        
    return avrgs

"""
Layman terms: iterates over all ratings for that joke and weights addition by how similar user a is to user b
Adapted so can be used with N neighbors and adjusted sum
"""
def weighted_sum(df_na, cases, avrgs, N = None, adj = False):
    df = df_na.copy().fillna(0) #since for computations we assume empty value is 0
    results = []
    curr_avrg = 0

    
    for case in cases: #cases is list of test cases [(user1, joke1),....]
        df_curr = df.copy() #copies as to not modif y original, r
        user = case[0]
        joke = case[1]
        df_curr.at[user, joke] = 0 #assigns that value to 0, as it can modify the similarity calculation below

        sim_ser = calc_sim(user, df_curr) #series of similartiies, excluding the user observation
        
        if adj:
            curr_avrg = df_na.drop(joke, axis = 1).iloc[user].mean(skipna = True)
            #this looks scary, but all it does is averages values of row of current user without current joke observation
        if N is None: #not neearest neighbors
            C = [i for i in range(df.shape[0]) if i != user]
            #makes list of C's s.t c \neq user
        else:
            C = calc_neighbors(sim_ser, N) #sim_vec[i] is similarity value between current userent user
            #C is list of indices for N most similar users
            sim_ser = sim_ser.loc[C] #limits series to only those N neighbors, important for average calculation
       
        sum = 0
        k = sim_ser.abs().sum() #normalizing factor, check documentation
        for c in C: #for each user in users
            sum += sim_ser[c] * (df_curr.at[c, joke] - avrgs[c]) #check documentation sim * (u(c,s) - avrg(c))

        res = curr_avrg + (sum / k) #check documnetation
        res = round(res, 2) #CAN COMMENT OUT, EASIER TO READ
        results.append(res)

    return results #returns list of predicted values alligning with index of test cases


"""
Calculates similarity, user is an index
returns series where each value is similarity to user and index is obviously that user id. Drops user as to not create issues for n-largest.
"""
def calc_sim(user, df): #pearson!
    user_ser = df.iloc[user]
    sims = []
    for i in range(df.shape[0]):
        sim = user_ser.corr(df.iloc[i])
        sims.append(sim)

    sims = pd.Series(sims)
    sims = sims.drop(user)
    return sims

def calc_neighbors(sims, N):
    return sims.nlargest(N).index


