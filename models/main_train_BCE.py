
import csv
import argparse as ap
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from models import FeedForward, FeedForwardNLayers
from metrics import metrics
from loss import FocalLoss

# load the data
data_features = 'data/egfr_erbB1_train_pca.csv'
data_labels = 'data/egfr_erbB1_train_pca_labels.csv'

train_data = np.loadtxt(data_features, delimiter=',')
train_labels = np.loadtxt(data_labels, delimiter=',')[:, np.newaxis]

# split to train and test
feat_train, feat_val, lbl_train, lbl_val = train_test_split(
        train_data[ 0:6053 ], train_labels[ 0:6053 ], test_size=0.2, random_state=20,
        stratify=train_labels[0:6053]
        )

# training function
def train(
        train_steps=4000, batch_size=100, learning_rate=0.001, loss_function='BCE', focal_gamma=2,
        hidden_layers=None
        ):
    if hidden_layers is not None:
        model = FeedForwardNLayers( hidden_layers )
    else:
        model = FeedForward()

    # initalize model parameters
    # for param in model.parameters():
    #     torch.nn.init.xavier_uniform(param)

    if torch.cuda.is_available():
        print( 'Cuda is available.' )
        model = model.cuda()

    optimizer = torch.optim.Adam( model.parameters(), lr=learning_rate )

    # choose loss function
    if loss_function == "BCE":
        loss_fn = F.binary_cross_entropy_with_logits

    elif loss_function == "focal":
        loss_fn = FocalLoss( focal_gamma )

    else:
        raise NotImplementedError( f"'{loss_function}' is not implemented" )

    best_model = None
    best_val_acc = 0
    for step in range( train_steps ):
        i = np.random.choice( feat_train.shape[ 0 ], size=batch_size, replace=False )
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
            print(
                    f'On step {step:6d}, train loss {tr_loss:0.4f}, train accuracy {tr_accuracy * 100:.2f}%, val loss {val_loss:0.4f}, val accuracy {val_accuracy * 100:.2f}%'
                    )

    print( 'Done training' )
    # torch.save( best_model, 'models/best_NN_model')
    return best_model


def prediction( model, data, label, loss_fn ):
    with torch.no_grad():
        logits = model( data.float() )
        loss = loss_fn( logits, label )
        label_pred = torch.sigmoid( logits ).round()
    return loss, label_pred


def test( model ):
    pred = [ ]
    model.eval()
    # model = torch.load('models/best_NN_model')
    x = torch.tensor( feat_val )
    x = x.float()
    print( x.shape )
    if torch.cuda.is_available():
        x.cuda()
    # x = torch.tensor(x.values)
    logits = model( x )
    # logits = torch.transpose(logits, 0, 1)
    print( logits )
    print( logits.shape )
    pred = torch.sigmoid(logits).round().detach().numpy()
    print( pred )

    return pred

#
def main():
   # Parameters to search
    param_grid = {
        'train_steps': [1000, 4000, 7000, 10000],
        'batch_size': [50, 100, 150],
        'learning_rate': [0.1, 0.01, 0.001],
        'loss_function': ['BCE'], # performing binary cross entropy loss training
        'focal_gamma': [1, 2, 3],
        'hidden_layers': [[100]*2, [100]*3, None],
    }

    # Feature search
    best_model = None
    best_val_acc = -1

    log_f = open('all_param_BCE.csv', 'w')
    fieldnames = ['steps', 'batch_size', 'learning_rate', 'loss', 'hidden_layers',
                  'accuracy', 'specificity', 'sensitivity', 'precision', 'recall', 'f1_score']
    logger = csv.DictWriter(log_f, fieldnames)
    logger.writeheader()

    # collection of nested loops to iterate through each possibility
    for steps in param_grid[ 'train_steps' ]:
        for bs in param_grid[ 'batch_size' ]:
            for lr in param_grid[ 'learning_rate' ]:
                for loss in param_grid[ 'loss_function' ]:
                    for hidden_layers in param_grid[ 'hidden_layers' ]:

                        # run training, testing, store the parameters
                        curr_model = train(
                            train_steps=steps, batch_size=bs, learning_rate=lr, loss_function=loss,
                            hidden_layers=hidden_layers
                            )
                        pred = test( curr_model )
                        curr_acc, curr_spec, curr_sens, curr_prec, curr_reca, curr_f1 = metrics( lbl_val, pred )
                        all_param = {'steps': steps,
                                    'batch_size': bs,
                                     'learning_rate' : lr,
                                     'loss' : loss,
                                     'hidden_layers' : hidden_layers,
                                    'accuracy' : curr_acc,
                                     'specificity' : curr_spec,
                                     'sensitivity' : curr_sens,
                                     'precision' : curr_prec,
                                     'recall' : curr_reca,
                                     'f1_score' : curr_f1
                                     }
                        logger.writerow(all_param)

                        # compare accuracy and store current best model
                        if (curr_acc > best_val_acc):
                            best_val_acc = curr_acc
                            best_model = curr_model
                            best_params = [ steps, bs, lr, loss, hidden_layers ]


    # printing out the best parameters and best achieved accuracy
    print('Best Parameters:', '\n', 'train_steps: ', best_params[ 0 ], '\n', 'batch_size: ',
        best_params[ 1 ], '\n', 'learning_rate: ', best_params[ 2 ], '\n', 'loss_function: ',
        best_params[ 3 ], '\n', 'hidden_layers', best_params[ 4 ])

    if best_model is not None:
        torch.save(best_model, 'models/NN-BCE-best.torch')
        log_f.close()


# main

if __name__ == "__main__":
    main()

