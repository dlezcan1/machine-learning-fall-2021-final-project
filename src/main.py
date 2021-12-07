import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from models import FeedForward, FeedForwardNLayers
from metrics import metrics
import torch
import torch.nn.functional as F
import argparse as ap
from loss import FocalLoss


# ADD ARGS FUNCTION
def get_args():
    p = ap.ArgumentParser()

    p.add_argument( "mode", choices=[ "train", "test" ], type=str )

    p.add_argument( "--data-dir", type=str, default="data/egfr_erbB1_train_pca.csv" )
    p.add_argument( "--label-dir", type=str, default="data/gfr_erbB1_train_pca_labels.csv" )
    p.add_argument( "--log-file", type=str, default="NN-egfr-logs.csv" )
    p.add_argument( "--model-save", type=str, default="models/NN.torch" )
    p.add_argument( "--predictions-file", type=str, default="NN-egfr-preds.txt" )

    # Hyperparameters
    p.add_argument( "--model", type=str, default="NN" )
    p.add_argument( "--train_steps", type=int, default="4000" )
    p.add_argument( "--batch-size", type=int, default=100 )
    p.add_argument( "--learning-rate", type=float, default=0.001 )
    p.add_argument( "--loss-function", type=str, default='BCE' )
    p.add_argument( '--focal-gamma', type=float, default=2 )  # chosen from paper
    p.add_argument("--hidden-layers", type=int, nargs="+", default=None)

    return p.parse_args()


# train_feat = pd.read_csv('egfr_erbB1_train_pca.csv')
# train_labels = pd.read_csv('egfr_erbB1_train_pca_labels.csv')

def train( args ):
    train_data = pd.read_csv( args.data_dir )
    train_labels = pd.read_csv( args.label_dir )
    # Split training data into training and validation datasets.
    feat_train, feat_val, lbl_train, lbl_val = train_test_split(
            train_data, train_labels, test_size=0.2, random_state=20,
            stratify=train_labels
            )

    if args.model == 'NN' and args.hidden_layers is None:
        model = FeedForward()
    elif args.model == "NN" and args.hidden_layers is not None:
        model = FeedForwardNLayers(args.hidden_layers)
    # else:
    #     model =

    if torch.cuda.is_available():
        print( 'Cuda is available.' )
        model = model.cuda()

    log_f = open( args.log_file, 'w' )
    fieldnames = [ 'step', 'train_loss', 'train_acc', 'val_loss', 'val_acc' ]
    logger = csv.DictWriter( log_f, fieldnames )
    logger.writeheader()

    optimizer = torch.optim.Adam( model.parameters(), lr=args.learning_rate )

    # choose loss function
    if args.loss_function == "BCE":
        loss_fn = F.binary_cross_entropy_with_logits

    elif args.loss_function == "focal":
        loss_fn = FocalLoss( args.focal_gamma )

    else:
        raise NotImplementedError(f"'{args.loss_function}' is not implemented")

    best_model = None
    best_val_acc = 0
    for step in range( args.steps ):
        i = np.random.choice( feat_train.shape[ 0 ], size=args.batch_size, replace=False )
        feat = torch.from_numpy( feat_train[ i ].astype( np.float32 ) )
        lbl = torch.from_numpy( lbl_train[ i ].astype( np.int ) )

        if torch.cuda.is_available():
            feat = feat.cuda()
            lbl = lbl.cuda()

        logits = model( feat )

        loss = loss_fn( logits, lbl )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            tr_loss, tr_prediction = prediction( model, feat, lbl )
            tr_accuracy, tr_specificity, tr_sensitivity, tr_precision, tr_recall, tr_f1_score = metrics(
                    tr_prediction, lbl_train
                    )
            val_loss, val_prediction = prediction( model, feat_val, lbl_val )
            val_accuracy, val_specificity, val_sensitivity, val_precision, val_recall, val_f1_score = metrics(
                    val_prediction,
                    lbl_val
                    )

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_model = model

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

            print( f'On step {step}, train loss {tr_loss}, val accuracy {val_accuracy}' )
            logger.writerow( step_metrics )
    log_f.close()
    print( 'Done training' )
    torch.save( best_model, args.save_file_name )


def prediction( model, data, label ):
    logits = model( data )
    loss = F.cross_entropy( logits, label )
    label_pred = torch.max( logits, 1 )[ 1 ]
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
    pred = [ ]
    model = torch.load( args.model_save )
    x = pd.read_csv( args.data_dir )
    if torch.cuda.is_available():
        x.cuda()
    logits = model( x )
    pred.append( torch.max( logits, 1 )[ 1 ].item() )
    print( 'Storing predictions at {args.pred_file}' )
    pred = np.array( pred )
    np.savetext( args.pred_file, pred, fmt='%d' )


if __name__ == '__main__':
    args = get_args()
    if args.mode == 'train':
        train( args )
    elif args.mode == 'test':
        test( args )
    else:
        print( 'Invalid mode.' )
