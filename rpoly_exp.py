
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression

from sklearn import metrics
import myutil as mu
import numpy as np

from sklearn.linear_model import RidgeClassifierCV

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from sklearn.feature_selection import SelectKBest,VarianceThreshold
from sklearn.feature_selection import f_classif, SelectFromModel
from sklearn.preprocessing import StandardScaler

from sktime.transformations.panel.rocket import Rocket, MiniRocket
from tsfresh.feature_extraction.feature_calculators import *

import random
import logging
import sys
from datetime import datetime
import argparse
#logging.basicConfig(stream=sys.stdout, level=25)
from sktime.datatypes._panel._convert import from_nested_to_2d_array
#from sktime.datatypes._panel._convert import *
#https://stackoverflow.com/questions/6881170/is-there-a-way-to-autogenerate-valid-arithmetic-expressions


class RandomPolynomialTS(BaseEstimator, TransformerMixin):            
    
    def __init__(self, num_kernels=100, func = ["maximum"], random_state=42, keep_origin=False, group_prob=0.5): 
        self.num_kernels = num_kernels
        self.random_state = random_state
        self.keep_origin = keep_origin
        self.group_prob = group_prob
        self.func = func

    def generate_kernel(self, minTerminal, maxTerminal, maxdepth, group_prob = 0.5, depth=0):

        GROUP_PROB = group_prob
        expr = ""
        CONST_PROB = 0.0 # it seems const doesn't help
        UNARY_PROB = 0.0 # it seems unary doesn't help

        if np.random.uniform() < UNARY_PROB:
            return np.random.choice(["np.cos","np.exp"]) + "(" + f'X_w[:,{np.random.randint(minTerminal, maxTerminal)}]' + ")"

        grouped = np.random.uniform()< GROUP_PROB    
        if grouped:
            expr += '('

        if depth < maxdepth and np.random.randint(0, maxdepth) > depth:
            expr += self.generate_kernel(minTerminal, maxTerminal, maxdepth, group_prob=group_prob, depth=depth + 1)
        else:
            if np.random.uniform() < CONST_PROB:
                expr += f'{np.random.uniform()}'
            else:
                expr += f'X_w[:,{np.random.randint(minTerminal, maxTerminal)}]'

        expr += np.random.choice([" * ", " - ", " + "])

        if depth < maxdepth and np.random.randint(0, maxdepth) > depth:
            expr += self.generate_kernel(minTerminal, maxTerminal, maxdepth, group_prob=group_prob, depth=depth + 1)
        else:
            if np.random.uniform() < CONST_PROB:
                expr += f'{np.random.uniform()}'
            else:
                expr += f'X_w[:,{np.random.randint(minTerminal, maxTerminal)}]'

        if grouped:
            expr += ')'
        return expr
    
    def compute_simple_features(self,X):
        # funcs = {
        #     "abs_energy": abs_energy,            
        #     "maximum": maximum,
        #     "absolute_maximum": absolute_maximum,
        #     "minimum": minimum,
        #     "absolute_sum_of_changes": absolute_sum_of_changes,   
            # kurtosis,
            # longest_strike_above_mean,
            # longest_strike_below_mean,
            # mean,
            # mean_abs_change,
            # mean_change,
            # mean_second_derivative_central,
            # median,
            # root_mean_square,
            # sample_entropy,
            # skewness,
            # standard_deviation,
            # sum_values,
            # variance            
        #}
        
        fts = []  
                
        for f in self.func:
            fts.append(np.apply_along_axis(lambda x: [globals()[f](x)], 1, X))
        
        return np.hstack(fts)
            
    def fit(self, X, y = None):   
        np.random.seed(self.random_state)
        input_length = X.shape[1]
        self.kernels = []   
        self.windows = []
        
        for i in range(self.num_kernels):
            
            window_size = np.random.randint(8,input_length)
            kernel = self.generate_kernel(0, window_size, 3, group_prob = self.group_prob)
            
            
            self.kernels.append(kernel)
            self.windows.append(window_size)
            
        return self    
    
    def transform(self, X, y = None):
        ts_length = X.shape[1]
        X_final = []
        for window_size, kernel in zip(self.windows, self.kernels):
            X_transform = np.zeros((X.shape[0],ts_length-window_size+1))
            for i in range(ts_length-window_size+1):    
                X_w = X[:,i:(i+window_size)]
                X_transform[:,i] = eval(kernel)
            X_final.append(self.compute_simple_features(X_transform))
            #X_final.append(X_transform.max(axis=1,keepdims=True))
            #X_final.append(X_transform.min(axis=1,keepdims=True))
            #X_final.append(X_transform.std(axis=1,keepdims=True))            
            #X_final.append(np.median(X_transform,axis=1,keepdims=True))  
            #X_final.append(np.percentile(X_transform, 75, axis=1, keepdims=True))
                      
        return np.hstack(X_final)
        










    

def test_rpoly_ts(ds, num_kernels=1000, func=["maximum"]):
    X_train,y_train,X_test,y_test = mu.load_uea_arff_data(ds)
    X_train = from_nested_to_2d_array(X_train).values
    X_test = from_nested_to_2d_array(X_test).values
    
    clf = Pipeline(
            [   #('feature_selection', SelectFromModel(RidgeClassifierCV(alphas=np.logspace(-5, 5, 10), normalize=True))),
                ('bespoke', RandomPolynomialTS(num_kernels=num_kernels,func=func)),
                #('normalizer', StandardScaler()),
                #('constantfilter', VarianceThreshold()),
                #('selectkbest', SelectKBest(k=7000)),
                #('model', LinearDiscriminantAnalysis())
                ('model', RidgeClassifierCV(alphas = np.logspace(-5, 5, 10)))
                ],
            verbose=False,
        )
    clf.fit(X_train, y_train)
    

    

    return [metrics.accuracy_score(y_test, clf.predict(X_test))]



if __name__ == "__main2__":
    print(test_rpoly_ts('Coffee'))

if __name__ == "__main__":    

    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--name', type=str,
                    help='Name of the experiment', default="rpoly")
    parser.add_argument('--desc', type=str,
                    help='Description of the experiment', default="An interesting experiment.")
    parser.add_argument('--path', type=str,
                    help='Folder to store the results', default="rpoly/")
    parser.add_argument('--nk', type=int,
                    help='Number of kernels', default=1000)   
    parser.add_argument('--func', type=str,
                    help='tsfresh func', default="maximum")    
    args = parser.parse_args()
    

    print(args.name)
    print(args.desc)
    print(args.path)
    print(args.nk)
    print(args.func)

    
    log_file = args.path + args.name + "_" + datetime.now().strftime("%y%m%d%H%M") + ".log"
    logging.basicConfig(filename=log_file,                                 
                                 format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                                 level=45,
                                 datefmt='%Y-%m-%d %H:%M:%S')    
           
    logging.log(50,args.desc)  

    funcs = []
    for f in args.func.split(','):
        funcs.append(f)  
    print(funcs)

    with open(args.path + args.name + ".csv",'w') as f:
        f.write("Dataset," + args.name + "\n")
        for ds in mu.uea_fixed_length_datasets():
            logging.log(50,"Experiment with Dataset: [" + ds + "]")
            #acc = channelselection_mrsqm(ds,args.nsax,args.nsfa,args.maxch)
            acc = test_rpoly_ts(ds,args.nk,funcs)
            #acc = ','.join(map(str,acc))
            logging.log(50,"Accuracy: [" + str(acc) + "]")
            f.write(ds + "," + str(acc) + "\n")
            

    mu.read_log_file(log_file,args.path + "elapsed/" + args.name + ".csv",write_elapsed=True,write_acc=False)