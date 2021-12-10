# JUST IMPORTS
from sklearn.ensemble import RandomForestClassifier
import csv
import argparse as ap
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.metrics import confusion_matrix


# loading the data
url_data = "https://raw.githubusercontent.com/dlezcan1/machine-learning-fall-2021-final-project/main/data/egfr_erbB1_train_pca.csv"
url_labels = "https://raw.githubusercontent.com/dlezcan1/machine-learning-fall-2021-final-project/main/data/egfr_erbB1_train_pca_labels.csv"
data = pd.read_csv(url_data, error_bad_lines=False)
labels = pd.read_csv(url_labels, error_bad_lines=False)

  
# Split training data into training and validation datasets.
feat_train, feat_val, lbl_train, lbl_val = train_test_split(data, labels, test_size=0.2, random_state=20, stratify=labels)


# Harrison's metrics function
def metrics(lbl, pred):
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    tn, fp, fn, tp = confusion_matrix(lbl, pred).ravel()
    accuracy = (tn + tp) / (tn + tp + fn + fp)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
  return accuracy, specificity, sensitivity, precision, recall, f1_score


# argument parsing
def get_args():
  p = ap.ArgumentParser()

  p.add_argument( "mode", choices=[ "train", "test" ], type=str )
    #p.add_argument( "--log-file", type=str, default="models/NN-egfr-logs.csv" )
  p.add_argument( "--model-save", type=str, default="models/" )
  p.add_argument( "--predictions-file", type=str, default="NN-egfr-preds.txt" )

    # Hyperparameters

  p.add_argument("--model", type=str, default="RFC")
  p.add_argument( "--number-estimators", type=int, default=10 )
  p.add_argument( "--max-features", type=str, default='auto' )
  p.add_argument("--max-depth", type=int, default=None)
  p.add_argument( "--min-samples-leaf", type=int, default=1 )
  p.add_argument( "--min-samples-split", type=int, default=2 )
  p.add_argument( "--bootstrap", type=bool, default=True )

  return p.parse_args()


# training function
def train( args ):
  
  # defining the model
  rf = RandomForestClassifier(bootstrap=args.bootstrap, max_depth=args.max_depth, max_features=args.max_features,
            min_samples_leaf=args.min_samples_leaf, min_samples_split=args.min_samples_split, n_estimators=args.number_estimators)

  # reshape labels
  labels_train = lbl_train.values.ravel()
  
  # fitting model
  rf.fit(feat_train, labels_train)

  # saving model
  filename = 'finalized_rfc_model.csv'
  pickle.dump(rf, open(filename, 'wb'))


# testing function
def test( args ): 

  # open model
  filename = 'finalized_rfc_model.csv'
  rf = pickle.load(open(filename, 'rb'))

  # storing predictions
  y_pred = rf.predict(feat_val)
  filename2 = 'finalized_rfc_pred.csv'
  pickle.dump(y_pred, open(filename2, 'wb'))

  # using metrics
  accuracy, specificity, sensitivity, precision, recall, f1_score = metrics(lbl_val, y_pred)
  print("Accuracy: ", accuracy, "\n", "Specificity: ", specificity, "\n", "Sensitivity: ", sensitivity,  "\n", "Precision: " , precision, "\n",  "Recall: ",  recall,  "\n", "F1 Score: ",  f1_score)


# main
args = get_args()
if args.mode == 'train':
  train( args )
elif args.mode == 'test':
  test( args )
else:
  print( 'Invalid mode.' )