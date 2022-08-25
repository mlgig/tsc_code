import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.io import arff
#from sktime.utils.data_io import *
from sktime.datasets import load_from_arff_to_dataframe

import os
from multiprocessing import Pool
import pandas as pd
import re
import datetime
import subprocess
import time
import sys


ds_dir = '/media/thalng/DATA/thachln/myworkdir/data/UCR_TS_Archive_2015/'
uea_dir = '/media/thalng/DATA/thachln/myworkdir/data/UEA_2018_univariate_arff/'

def read_data(input_file):    
    if input_file.lower().endswith(".csv"):
        train_data = np.genfromtxt(input_file,delimiter=',')
        X = train_data[:,1:]
        y = train_data[:,0]
    elif input_file.lower().endswith(".txt"): 
        dt = np.loadtxt(input_file)
        y, X = dt[:, 0].astype(np.int), dt[:, 1:]
    elif input_file.lower().endswith(".arff"):        
        X,y = load_from_arff_to_dataframe(input_file)



    # test_data = np.genfromtxt(test_file,delimiter=',')
    # X_test = test_data[:,1:]
    # y_test = test_data[:,0]

    return X, y

def train_and_test_with_logistic_regression(train_x, train_y, test_x, test_y):
    clf = LogisticRegression(solver='newton-cg',multi_class = 'multinomial', class_weight='balanced', max_iter=1000).fit(train_x, train_y)
    # clf = LogisticRegression(solver='liblinear', class_weight='balanced').fit(train_x, train_y)
    predicted = clf.predict(test_x)
    return metrics.accuracy_score(test_y, predicted)

# def read_arff(arff_file):
#     data = arff.loadarff(arff_file)
#     X = []
#     y = []
#     for ts in data[0]:
#         ts_as_list = ts.tolist()
#         X.append(list(ts_as_list[:-1]))
#         y.append(ts_as_list[-1])
    
#     return X,y

def to_feature_space(sequences, mr_seqs):
    full_fm = []

    for rep, seq_features in zip(mr_seqs, sequences):            
        fm = np.zeros((len(rep), len(seq_features)))
        for i,s in enumerate(rep):
            for j,f in enumerate(seq_features):
                # fm[i,j] = convoluted_features(f,s)
                if f in s:
                    fm[i,j] = 1
        full_fm.append(fm)

  
    full_fm = np.hstack(full_fm)
    return full_fm

def ucr_2015_datasets():    
    ucr_list = [ds for ds in os.listdir(ds_dir)]
    return ucr_list
    # for ds in ucr_list:
    #     train_file = ds_dir + ds + '/' + ds + '_TRAIN' 
    #     test_file = ds_dir + ds + '/' + ds + '_TEST' 
    #     yield ds, train_file,test_file
def uea_2018_datasets():
    uea_list = []
    for ds in os.listdir(uea_dir):
        if os.path.isdir(uea_dir + ds): # check if it's a folder
            if any(fname.endswith('.arff') for fname in os.listdir(uea_dir + ds)): #check if the folder contains arff file
                uea_list.append(ds)
    return uea_list

def get_ucr_path(ds):
    # ds_dir = '/home/thachln/myworkdir/UCR_TS_Archive_2015/'
    train_file = ds_dir + ds + '/' + ds + '_TRAIN' 
    test_file = ds_dir + ds + '/' + ds + '_TEST'
    return train_file,test_file

def get_uea_path(ds):
    train_file = uea_dir + ds + '/' + ds + '_TRAIN.arff' 
    test_file = uea_dir + ds + '/' + ds + '_TEST.arff'
    return train_file,test_file


def get_ucr_data(ds):
    train_file,test_file = get_ucr_path(ds)
    train_x, train_y = read_data(train_file)
    test_x, test_y = read_data(test_file)

    return train_x, train_y, test_x, test_y

def Coffee_train_and_test():
    ds_dir = '/home/thachln/myworkdir/UCR_TS_Archive_2015/'
    ds = 'Coffee'
    train_file = ds_dir + ds + '/' + ds + '_TRAIN' 
    test_file = ds_dir + ds + '/' + ds + '_TEST' 
    return [[ds, train_file,test_file]]


def parallel_exp(func, pars, n_workers):
    p = Pool(n_workers)
    return p.map(func, pars)

def parallel_ma_exp(func, pars, n_workers):
    p = Pool(n_workers)
    return p.starmap(func, pars)

def SITS_train_and_test(fold):
    ds_dir = '/home/thachln/myworkdir/data/1M-TSC-SITS_2006_NDVI_C/'
    ds_prefix = 'SITS1M_fold'    
    ds = ds_prefix + str(fold)
    train_file = ds_dir + ds + '/' + ds + '_TRAIN.csv' 
    test_file = ds_dir + ds + '/' + ds + '_TEST.csv' 

    return train_file, test_file

def np_array_to_df(nparray):
    X = pd.DataFrame()
    X['dim_0'] = [pd.Series(x) for x in nparray]
    return X

def load_uea_arff_data(ds, isMultivariate=False):
    if isMultivariate:    
        train_file = "/home/thachln/myworkdir/data/Multivariate_arff/" + ds + "/" + ds + "_TRAIN.arff"
        test_file = "/home/thachln/myworkdir/data/Multivariate_arff/" + ds + "/" + ds + "_TEST.arff"
    else:
        train_file, test_file = get_uea_path(ds)
    train_x, train_y = read_data(train_file)
    test_x, test_y = read_data(test_file)

    return train_x, train_y, test_x, test_y

def uea_variable_length_datasets(recheck = False):
    rt = ['PLAID', 'GesturePebbleZ2', 'GestureMidAirD3', 'DodgerLoopWeekend', 'MelbournePedestrian', 
        'DodgerLoopDay', 'GestureMidAirD2', 'PickupGestureWiimoteZ', 'AllGestureWiimoteX', 'ShakeGestureWiimoteZ', 
        'GesturePebbleZ1', 'AllGestureWiimoteZ', 'DodgerLoopGame', 'GestureMidAirD1', 'AllGestureWiimoteY']
    if recheck:
        rt = []
        for ds in uea_2018_datasets():
            train_x, train_y, test_x, test_y = load_uea_arff_data(ds)
            # l = len(train_x.iloc[0, 0])
            for a in train_x.iloc[:, 0]:                
                if np.sum(np.isnan(a)) > 0:
                    rt.append(ds)
                    break
    return rt

def uea_fixed_length_datasets():
    return list(filter(lambda a: a not in uea_variable_length_datasets(), uea_2018_datasets()))

def uea_spectro_datasets():
    return ["Beef","Coffee","Ham","Meat","OliveOil","EthanolLevel"]

def uea_small_multivariate_dataset():
    return ["ArticularyWordRecognition","AtrialFibrillation","BasicMotions","Cricket","ERing","EigenWorms","Epilepsy","EthanolConcentration",'FingerMovements',
            "HandMovementDirection","Handwriting","LSST","Libras","NATOPS","PenDigits","PhonemeSpectra",
            "RacketSports","SelfRegulationSCP1","SelfRegulationSCP2","StandWalkJump","UWaveGestureLibrary"]

def uea_multivariate_dataset():
    return ["ArticularyWordRecognition","AtrialFibrillation","BasicMotions","Cricket","ERing","EigenWorms","Epilepsy","EthanolConcentration",'FingerMovements',
            "HandMovementDirection","Handwriting","LSST","Libras","NATOPS","PenDigits","PhonemeSpectra",
            "RacketSports","SelfRegulationSCP1","SelfRegulationSCP2","StandWalkJump","UWaveGestureLibrary","MotorImagery","Heartbeat","FaceDetection","PEMS-SF","DuckDuckGeese"]
def read_reps_from_file(inputf):
    last_cfg = None
    mr_seqs = []
    rep = []
    i = 0
    for l in open(inputf,"r"):
        i += 1
        l_splitted = bytes(l,'utf-8').split(b" ")
        cfg = l_splitted[0]
        seq = b" ".join(l_splitted[2:])
        if cfg == last_cfg:
            rep.append(seq)
        else:
            last_cfg = cfg
            if rep:
                mr_seqs.append(rep)
            rep = [seq]
    if rep:
        mr_seqs.append(rep)    
    return mr_seqs

def load_uea_sfa_reps(ds):
    # sfa_dir = "/home/thachln/myworkdir/experiments/sfa_x2_uea/"
    # return read_reps_from_file(sfa_dir + ds + '/sfa.train'), read_reps_from_file(sfa_dir + ds + '/sfa.test')
    sfa_dir = "/home/thachln/myworkdir/FastSFA/boss/out/"
    return read_reps_from_file(sfa_dir + ds + 'train'), read_reps_from_file(sfa_dir + ds + '.test')



def read_log_file(filename,outfile,write_elapsed=False,write_acc=True):
    f = open(outfile,'w')

    time_pattern = re.compile('\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3}')
    vl_pattern = re.compile('\[.*\]')
    for line in open(filename,'r'):
        if "Dataset" in line: # beginning of experiment
            ds = vl_pattern.search(line).group()[1:-1]
            timestamp = time_pattern.search(line).group()            
            time_begin = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
        elif "Accuracy" in line: # end of experiment            
            timestamp = time_pattern.search(line).group()
            time_end = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
            elapsed = (time_end - time_begin).total_seconds()
            acc = vl_pattern.search(line).group()[1:-1]
            f.write(ds)
            if write_acc:                
                f.write(',' + acc)
            if write_elapsed:
                f.write(',' + str(elapsed))
            f.write('\n')
    f.close()



def sfa_transform(itrain, itest, otrain, otest):
    subprocess.call(['java', '-jar', 'TestSFA-all.jar', itrain, itest, otrain, otest])

def time_sfa():
    f = open('sfatime.csv','w')
    for ds in uea_fixed_length_datasets():
        begin = time.time()
        sfa_transform(uea_dir + '/' + ds + '/' + ds + '_TRAIN.txt',uea_dir + '/' + ds + '/' + ds + '_TEST.txt','sfa.train','sfa.test')
        f.write(ds + ',' + str(time.time() - begin) + '\n')        
    f.close()

def load_CMJ():
    train_file = '/home/thachln/myworkdir/mrsqm/example/data/CMJ/CMJ_TRAIN'
    test_file = '/home/thachln/myworkdir/mrsqm/example/data/CMJ/CMJ_TEST'
    X_train, y_train = load_from_ucr_tsv_to_dataframe(train_file)
    X_test, y_test = load_from_ucr_tsv_to_dataframe(test_file)

    return X_train, y_train, X_test, y_test

def load_origin_crop():
    train_file = '/home/thachln/myworkdir/data/1M-TSC-SITS_2006_NDVI_C/SITS1M_fold1/SITS1M_fold1_TRAIN.csv'
    test_file = '/home/thachln/myworkdir/data/1M-TSC-SITS_2006_NDVI_C/SITS1M_fold1/SITS1M_fold1_TEST.csv'



    X_train, y_train = read_data(train_file)
    X_test, y_test = read_data(test_file)

    return X_train, y_train, X_test, y_test

def load_DuckAndGeese():
    train_file = '/home/thachln/myworkdir/data/DucksAndGeese/DucksAndGeese_TRAIN.arff'
    test_file = '/home/thachln/myworkdir/data/DucksAndGeese/DucksAndGeese_TEST.arff'



    X_train, y_train = read_data(train_file)
    X_test, y_test = read_data(test_file)

    return X_train, y_train, X_test, y_test

def write_csv(X,y,target_file):
    y_col = np.array([y])
    N = X.shape[1]
    np.savetxt(target_file,np.hstack((y_col.T,X)), fmt=','.join(['%i'] + ['%.4f']*N), delimiter = ",")



if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
 
    #print(read_reps_from_file("/home/thachln/myworkdir/data/UEA_2018_univariate_arff/MoteStrain/sfa.train"))
    read_log_file(input_file,output_file,write_elapsed=True,write_acc=False)
    

'''
Remove ROCKET cache
"rm -r $VIRTUAL_ENV/lib/python3.6/site-packages/sktime/transformations/panel/rocket/__pycache__/*"
'''