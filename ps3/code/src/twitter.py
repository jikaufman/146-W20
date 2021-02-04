"""
Author      : Yi-Chieh Wu, Sriram Sankararman
Description : Twitter
"""

from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.
    
    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """
    
    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return
    
    np.savetxt(outfile, vec)    


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        current_index = 0
        extracted_words = extract_words(fid.read())
        for i in range(len(extracted_words)):
            if extracted_words[i] not in word_list:
                word_list[extracted_words[i]] = current_index
                current_index += 1
        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        lines = fid.readlines()
        for i in range(len(lines)):
            tweet = extract_words(lines[i])
            for j in range(len(tweet)):
                feature_matrix[i][word_list[tweet[j]]] = 1
        ### ========== TODO : END ========== ###
        
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    value = 0
    if metric == 'accuracy':
        value = metrics.accuracy_score(y_true, y_label)
    elif metric == 'f1_score':
        value = metrics.f1_score(y_true, y_label)
    elif metric == 'auroc':
        value = metrics.roc_auc_score(y_true, y_label)
    elif metric == 'precision':
        value = metrics.precision_score(y_true, y_label)
    elif metric == 'sensitivity':
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_label).ravel()
        value = tp / float(tp + fn)
    elif metric == 'specificity':
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_label).ravel()
        value = tn / float(fp + tn)

    return value
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance   

    sum_score = 0
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        y_pred = clf.decision_function(X_test)
        sum_score += performance(y_test, y_pred, metric=metric)
    return sum_score / float(5)
    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ':'
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2c: select optimal hyperparameter using cross-validation

    C_optimal = 0.0
    C_optimal_val = 0.0

    for c in C_range:
        clf = SVC(kernel='linear', C=c)
        fjfj = cv_performance(clf, X, y, kf, metric)
        print "C=" + str(c) + ": " + str(fjfj)
        if fjfj > C_optimal_val:
            C_optimal_val = fjfj
            C_optimal = c

    return C_optimal
    ### ========== TODO : END ========== ###


def select_param_rbf(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """
    
    print 'RBF SVM Hyperparameter Selection based on ' + str(metric) + ':'
    
    ### ========== TODO : START ========== ###
    # part 3b: create grid, then select optimal hyperparameters using cross-validation

    C_range = 10.0 ** np.arange(-3, 3)
    gamma_range = 10.0 ** np.arange(-3, 3)

    best_cv_performance = -1
    best_c = -1
    best_gamma = -1
    for c in C_range:
        for gamma in gamma_range:
            clf = SVC(kernel="rbf", gamma=gamma, C=c)
            perf = cv_performance(clf=clf, X=X, y=y, kf=kf, metric=metric)


            if perf > best_cv_performance:
                best_cv_performance = perf
                best_c = c
                best_gamma = gamma

    return best_c, best_gamma, best_cv_performance
    ### ========== TODO : END ========== ###


def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 4b: return performance on test data by first computing predictions and then calling performance

    score = 0        
    return score
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the tweets and its labels   
    dictionary = extract_dictionary('../data/tweets.txt')

    
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    y = read_vector_file('../data/labels.txt')
    
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    
    
    ### ========== TODO : START ========== ###
    # part 1c: split data into training (training + cross-validation) and testing set

    train_X, test_X = X[:560], X[560:]
    train_y, test_y = y[:560], y[560:]

    # part 2b: create stratified folds (5-fold CV)
    
    clf = SVC()

    kf = StratifiedKFold(n_splits=5)

    #print(cv_performance(clf, train_data_X, train_data_y, kf, metric="accuracy"))

    # part 2d: for each metric, select optimal hyperparameter for linear-kernel SVM using CV


    """

    for met in metric_list:
        k = select_param_linear(train_X, train_y, kf, metric=met)
        print "Optimal parameter: " + str(k)

    """

    best_linear_acc_c = select_param_linear(train_X, train_y, kf, metric='accuracy')

    # part 3c: for each metric, select optimal hyperparameter for RBF-SVM using CV

    """

    for met in metric_list:
        best_rbf_c, best_rbf_gamma, rbf_perf = select_param_rbf(X=train_X, y=train_y, kf=kf, metric=met)

        if metric == "accuracy":
            best_rbf_acc_c = best_rbf_c
            best_rbf_acc_gamma = best_rbf_gamma

        print "{} : {}, {}, {}".format(met, rbf_perf, best_rbf_c, best_rbf_gamma)

    """

    best_rbf_acc_c, best_rbf_acc_gamma, foo = select_param_rbf(X=train_X, y=train_y, kf=kf, metric="accuracy")

    # part 4a: train linear- and RBF-kernel SVMs with selected hyperparameters

    linear_svm_clf = SVC(kernel="linear", C=best_linear_acc_c)
    linear_svm_clf.fit(train_X, train_y)

    rbf_svm_clf = SVC(kernel="linear", gamma=best_rbf_acc_c, C=best_rbf_acc_gamma)
    rbf_svm_clf.fit(train_X, train_y)
    
    # part 4c: report performance on test data

    for metric in metric_list:
        linear_perf, rbf_perf = performance_test(linear_svm_clf, test_X, test_y, metric), performance_test(rbf_svm_clf, test_X, test_y, metric)
        print("{}:".format(metric))
        print("Linear SVM : {}".format(linear_perf))
        print("RBF SVM : {}".format(rbf_perf))
    
    ### ========== TODO : END ========== ###
    
    
if __name__ == "__main__" :
    main()
