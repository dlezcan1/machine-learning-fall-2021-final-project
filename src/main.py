import copy
import csv
import argparse as ap
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# custom imports
from metrics import metrics
from loss import FocalLoss
from models import FeedForward, FeedForwardNLayers


# ADD ARGS FUNCTION
def get_args():
    p = ap.ArgumentParser()

    p.add_argument( "mode", choices=[ "train", "test" ], type=str )

    p.add_argument( "--data-dir", type=str, default="data/egfr_erbB1_train_pca.csv" )
    p.add_argument( "--label-dir", type=str, default="data/egfr_erbB1_train_pca_labels.csv" )
    p.add_argument( "--log-file", type=str, default="models/NN-egfr-logs.csv" )
    p.add_argument( "--model-save", type=str, default="models/NN.torch" )
    p.add_argument( "--predictions-file", type=str, default="NN-egfr-preds.txt" )

    # Hyperparameters
    p.add_argument( "--model", type=str, default="NN" )
    p.add_argument( "--train-steps", type=int, default=4000 )
    p.add_argument( "--batch-size", type=int, default=100 )
    p.add_argument( "--learning-rate", type=float, default=0.001 )
    p.add_argument( "--loss-function", type=str, default='BCE' )
    p.add_argument( '--focal-gamma', type=float, default=2 )  # chosen from paper
    p.add_argument( "--hidden-layers", type=int, nargs="+", default=None )

    return p.parse_args()


# train_feat = pd.read_csv('egfr_erbB1_train_pca.csv')
# train_labels = pd.read_csv('egfr_erbB1_train_pca_labels.csv')

def train( args ):
    train_data = np.loadtxt( args.data_dir, delimiter=',' )
    train_labels = np.loadtxt( args.label_dir, delimiter=',' )[ :, np.newaxis ]

    # Split training data into training and validation datasets.
    feat_train, feat_val, lbl_train, lbl_val = train_test_split(
            train_data, train_labels, test_size=0.2, random_state=20,
            stratify=train_labels
            )

    if args.model == 'NN' and args.hidden_layers is None:
        model = FeedForward()
    elif args.model == "NN" and args.hidden_layers is not None:
        model = FeedForwardNLayers( args.hidden_layers )
    else:
        raise NotImplementedError( f"'{args.model}' is not implemented." )

    # initalize model parameters
    # for param in model.parameters():
    #     torch.nn.init.xavier_uniform(param)

    if torch.cuda.is_available():
        print( 'Cuda is available.' )
        model = model.cuda()

    log_f = open( args.log_file, 'w' )
    fieldnames = [ 'step', 'train_loss', 'train_accuracy', 'train_specificity', 'train_sensitivity',
                   'train_precision', 'train_recall', 'train_f1_score', 'val_loss', 'val_accuracy',
                   'val_specificity', 'val_sensitivity', 'val_precision', 'val_recall',
                   'val_f1_score' ]
    logger = csv.DictWriter( log_f, fieldnames )
    logger.writeheader()

    optimizer = torch.optim.Adam( model.parameters(), lr=args.learning_rate )

    # choose loss function
    if args.loss_function == "BCE":
        loss_fn = F.binary_cross_entropy_with_logits

    elif args.loss_function == "focal":
        loss_fn = FocalLoss( args.focal_gamma )

    else:
        raise NotImplementedError( f"'{args.loss_function}' is not implemented" )

    best_model = None
    best_val_acc = 0
    for step in range( args.train_steps ):
        i = np.random.choice( feat_train.shape[ 0 ], size=args.batch_size, replace=False )
        feat = torch.from_numpy( feat_train[ i ] ).float()
        lbl = torch.from_numpy( lbl_train[ i ] ).float()

        if torch.cuda.is_available():
            feat = feat.cuda()
            lbl = lbl.cuda()

        logits = model( feat )

        loss = loss_fn( logits, lbl )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            tr_loss, tr_prediction = prediction( model, feat, lbl, loss_fn )
            tr_accuracy, tr_specificity, tr_sensitivity, tr_precision, tr_recall, tr_f1_score = metrics(
                    lbl, tr_prediction
                    )
            val_loss, val_prediction = prediction(
                    model, torch.from_numpy( feat_val ), torch.from_numpy( lbl_val ), loss_fn
                    )
            val_accuracy, val_specificity, val_sensitivity, val_precision, val_recall, val_f1_score = metrics(
                    lbl_val, val_prediction
                    )

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_model = copy.deepcopy(model)

            step_metrics = {
                    'step'             : step,
                    'train_loss'       : tr_loss.item(),
                    'train_accuracy'   : tr_accuracy,
                    'train_specificity': tr_specificity,
                    'train_sensitivity': tr_sensitivity,
                    'train_precision'  : tr_precision,
                    'train_recall'     : tr_recall,
                    'train_f1_score'   : tr_f1_score,
                    'val_loss'         : val_loss.item(),
                    'val_accuracy'     : val_accuracy,
                    'val_specificity'  : val_specificity,
                    'val_sensitivity'  : val_sensitivity,
                    'val_precision'    : val_precision,
                    'val_recall'       : val_recall,
                    'val_f1_score'     : val_f1_score,
                    }

            print(
                    f'On step {step:6d}, train loss {tr_loss:0.4f}, train accuracy {tr_accuracy * 100:.2f}%, val loss {val_loss:0.4f}, val accuracy {val_accuracy * 100:.2f}%'
                    )
            logger.writerow( step_metrics )
    log_f.close()
    print( f'Done training with accuracy: {best_val_acc*100}%' )
    torch.save( best_model, args.model_save )


def prediction( model, data, label, loss_fn ):
    with torch.no_grad():
        logits = model( data.float() )
        loss = loss_fn( logits, label )
        label_pred = torch.sigmoid(logits).round()
    return loss, label_pred


# Might not be necessary to calculate metrics at every 100 steps - just at the end.
# def metrics(lbl, pred):
#     tn, fp, fn, tp = confusion_matrix(lbl, pred)
#     accuracy = (tn + tp) / (tn + tp + fn + fp)
#     specificity = tn / (tn + fp)
#     sensitivity = tp / (tp + fn)
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     f1_score = (2 * precision * recall) / (precision + recall)
#     return accuracy, specificity, sensitivity, precision, recall, f1_score

def test( args ):
    model = torch.load( args.model_save )
    x = pd.read_csv( args.data_dir )
    if torch.cuda.is_available():
        x.cuda()
    logits = model( x )
    pred = torch.sigmoid(logits).round().to_numpy()
    print( 'Storing predictions at {args.pred_file}' )
    np.savetext( args.pred_file, pred, fmt='%d' )


if __name__ == '__main__':
    args = get_args()
    if args.mode == 'train':
        train( args )
    elif args.mode == 'test':
        test( args )
    else:
        print( 'Invalid mode.' )
