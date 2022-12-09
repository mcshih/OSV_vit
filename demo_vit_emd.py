import sys
import json
from PIL import Image
import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import timm
from model import ViT_for_OSV, ViT_for_OSV_v2, ViT_for_OSV_v3, ViT_for_OSV_emd
from sig_dataloader import SigDataset_BH as SigDataset_BH_v1
from sig_dataloader import SigDataset
from sig_dataloader_v2 import SigDataset_BH as SigDataset_BH_v2
from module.loss import ContrastiveLoss
from models_emd.Network import DeepEMD

from sklearn.metrics import auc
# REPRODUCIBILITY
#torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4, help='input batch size') # train: 4, test: 16
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to ARC')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--name', default="demo", help='Custom name for this configuration. Needed for saving'
                                                 ' model checkpoints in a separate folder.')
parser.add_argument('--load', default=None, help='the model to load from. Start fresh if not specified.')
parser.add_argument('--data', type=str, default="./../ChiSig", help='data path')
parser.add_argument('--data_mode', type=str, default="normalized", help='data path') # [normalized, cropped, centered, left]
parser.add_argument('--test_only', action='store_true', help='test mode')
parser.add_argument('--shift', action='store_true', help='shift during train/test')
parser.add_argument('--model_type', type=str, default="v1", help='model type') # v1: only vit, v2: vit + HelixTransformer, v3: vit + CPD
###Loss###
parser.add_argument('--loss', type=str, default="con", help='select loss') #['bce', 'con']

parser.add_argument('--comment', type=str, default="", help='some note')

###EMD###
parser.add_argument("--norm", type=str, default='center')
parser.add_argument("--metric", type=str, default='cosine')
parser.add_argument('--solver', type=str, default='opencv')
parser.add_argument('--temperature', type=float, default=1.0)

parser.add_argument('--distribution', type=str, default='v1') # v1, v2, uniform

opt = parser.parse_args()

def compute_accuracy_roc(predictions, labels, step=None):
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)
    if step is None:
        step = 0.00005

    max_acc, min_frr, min_far = 0.0, 1.0, 1.0
    min_dif = 1.0
    d_optimal = 0.0
    tpr_arr, fpr_arr, far_arr, frr_arr, d_arr = [], [], [], [], []
    for d in np.arange(dmin, dmax + step, step):
        idx1 = predictions.ravel() <= d     # pred = 1
        idx2 = predictions.ravel() > d      # pred = 0

        tpr = float(np.sum(labels[idx1] == 1)) / nsame
        tnr = float(np.sum(labels[idx2] == 0)) / ndiff

        frr = float(np.sum(labels[idx2] == 1)) / nsame
        far = float(np.sum(labels[idx1] == 0)) / ndiff

        tpr_arr.append(tpr)
        far_arr.append(far)
        frr_arr.append(frr)
        d_arr.append(d)

        acc = 0.5 * (tpr + tnr)
        
        # print(f"Threshold = {d} | Accuracy = {acc:.4f}")

        if acc > max_acc:
            max_acc = acc
            d_optimal = d
            
            # FRR, FAR metrics
            min_frr = float(np.sum(labels[idx2] == 1)) / nsame
            min_far = float(np.sum(labels[idx1] == 0)) / ndiff
        
        if abs(far-frr) < min_dif:
            min_dif = abs(far-frr)
            d_optimal_diff = d
            
            # FRR, FAR metrics
            min_dif_frr = float(np.sum(labels[idx2] == 1)) / nsame
            min_dif_far = float(np.sum(labels[idx1] == 0)) / ndiff
            
    print("EER: {} @{}".format((min_dif_frr+min_dif_far)/2.0, d_optimal_diff))
    metrics = {"best_acc" : max_acc, "best_frr" : min_frr, "best_far" : min_far, "tpr_arr" : tpr_arr, "far_arr" : far_arr, "frr_arr" : frr_arr, "d_arr": d_arr}
    return metrics, d_optimal

def plot_roc(tpr, fpr, fname):
    assert len(tpr) == len(fpr)
    plt.plot(fpr, tpr, marker='.')
    plt.plot(fpr, fpr, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(f"./ROC_{fname}.png", dpi=300)

def find_eer(far, frr, thresholds, fname):
    plt.plot(thresholds, far, marker = 'o',label = 'far')
    plt.plot(thresholds, frr, marker = 'o',label = 'frr')
    plt.legend()
    plt.xlabel('thresh')
    plt.ylabel('far/frr')
    plt.title('find eer')
    plt.savefig(f"./EER_{fname}.png")
    plt.close()

def get_pct_accuracy(pred: Variable, target, path=None) -> int:
    if opt.loss == 'con':
        hard_pred = (pred < 0.5).int()
    elif opt.loss == 'bce':
        hard_pred = (pred > 0.5).int()
    else:
        return NotImplementedError
    #hard_pred = (pred > 0.5).int()
    #hard_pred = (pred < 0.5).int()
    #correct = (hard_pred == target).sum().data[0]
    correct = (hard_pred == target).sum().data
    accuracy = float(correct) / target.size()[0]
    '''
    if accuracy != 1 and path is not None:
        f = open('wrong_result.txt', 'a')
        f.write(str(pred.tolist()))
        f.write(str(path)+'\n')
        f.close()
    '''
    accuracy = int(accuracy * 100)
    return accuracy

def train(opt):

    if 'BHSig260' in opt.data:
        sigdataset_train = SigDataset_BH_v1(opt, opt.data, train=True, image_size=opt.imageSize, mode=opt.data_mode)
        sigdataset_test = SigDataset_BH_v1(opt, opt.data, train=False, image_size=opt.imageSize, mode=opt.data_mode)
        #sigdataset_test = SigDataset_BH_v2(opt, opt.data, train=False, image_size=opt.imageSize, mode=opt.data_mode)
    elif 'ChiSig' in opt.data:
        sigdataset_train = SigDataset(opt.data, train=True, image_size=opt.imageSize)
        sigdataset_test = SigDataset(opt.data, train=False, image_size=opt.imageSize)
    else:
        print('not implement')
        return NotImplementedError

    train_loader = DataLoader(sigdataset_train, batch_size=opt.batchSize, shuffle=True)
    test_loader = DataLoader(sigdataset_test, batch_size=opt.batchSize, shuffle=False)

    # make directory for storing models.
    models_path = os.path.join("saved_models", opt.name)
    os.makedirs(models_path, exist_ok=True)
    with open(os.path.join(models_path, 'args.txt'),'w') as f:
        f.write(' '.join(str(x) for x in sys.argv))
        json.dump(opt.__dict__,f,indent=4)
    
    if opt.model_type == 'v1':
        model = ViT_for_OSV()
    elif opt.model_type == 'v2':
        model = ViT_for_OSV_v2()
    elif opt.model_type == 'v3':
        model = ViT_for_OSV_v3(opt)
    elif opt.model_type == 'emd':
        model = ViT_for_OSV_emd(opt)
    else:
        return NotImplementedError
    model.cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(os.path.join(models_path, opt.load)))
    
    if opt.loss == 'con':
        bce = ContrastiveLoss()
    elif opt.loss == 'bce':
        bce = torch.nn.BCELoss()
    else:
        return NotImplementedError
    optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.lr, weight_decay=1e-4)
    # torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    
    best_validation_loss = None
    saving_threshold = 1.02
    last_saved = datetime.utcnow()
    save_every = timedelta(hours=2)
    
    #x = torch.randn(1, 3, 224, 224)
    #output = model(x)
    #print(output.shape)

    period = 0
    test_period = 2
    for epoch in range(300): # 400
        training_loss = 0.0
        train_acc = val_acc = 0.0
        for X, Y in tqdm(train_loader):
            model.train()
            X = X.view(-1,2,opt.imageSize,opt.imageSize)
            Y = Y.view(-1,1)
            #print(X.shape, Y.shape)
            X = X.cuda()
            Y = Y.cuda()
            pred = model(X)
            if isinstance(pred, tuple):
                loss_0 = bce(pred[0], Y.float())
                loss_1 = bce(pred[1], Y.float())
                loss = loss_0 + loss_1
            else:
                loss = bce(pred, Y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.data
            if isinstance(pred, tuple):
                train_acc += get_pct_accuracy(pred[0], Y)
            else:
                train_acc += get_pct_accuracy(pred, Y)

        if (period) % test_period == 0:
            period = 0
            validation_loss = 0.0
            for X_val, Y_val in tqdm(test_loader):
                model.eval()
                # validate your model
                X_val = X_val.view(-1,2,opt.imageSize,opt.imageSize).cuda()
                Y_val = Y_val.view(-1,1).cuda()

                pred_val = model(X_val)

                if isinstance(pred_val, tuple):
                    loss_0 = bce(pred_val[0], Y_val.float())
                    loss_1 = bce(pred_val[1], Y_val.float())
                    loss_val = loss_0 + loss_1
                else:
                    loss_val = bce(pred_val, Y_val.float())
                #loss_val = bce(pred_val, Y_val.float())
                validation_loss += loss_val.data
                if isinstance(pred_val, tuple):
                    val_acc += get_pct_accuracy(pred_val[0], Y_val)
                else:
                    val_acc += get_pct_accuracy(pred_val, Y_val)
                
            training_loss /= (float)(len(train_loader))
            validation_loss /= (float)(len(test_loader))
            train_acc /= (float)(len(train_loader))
            val_acc /= (float)(len(test_loader))

            print("Iteration: {} \t Train: Acc={}%, Loss={} \t\t Validation: Acc={}%, Loss={}".format(
            epoch, train_acc, training_loss, val_acc, validation_loss
            ))

            if best_validation_loss is None:
                best_validation_loss = validation_loss

            if best_validation_loss > (saving_threshold * validation_loss):
                print("Significantly improved validation loss from {} --> {}. Saving...".format(
                    best_validation_loss, validation_loss
                ))
                model.save_to_file(os.path.join(models_path, str(epoch)+str(validation_loss)))
                best_validation_loss = validation_loss
                last_saved = datetime.utcnow()

            #if last_saved + save_every < datetime.utcnow():
            #    print("It's been too long since we last saved the model. Saving...")
            #    model.save_to_file(os.path.join(models_path, str(validation_loss)))
            #    last_saved = datetime.utcnow()
                
        training_loss = 0.0
        train_acc = val_acc = 0.0
        period += 1

def test(opt):
    if 'BHSig260' in opt.data:
        #sigdataset_test = SigDataset_BH_v2(opt, opt.data, train=False, image_size=opt.imageSize, mode=opt.data_mode)
        sigdataset_test = SigDataset_BH_v1(opt, opt.data, train=False, image_size=opt.imageSize, mode=opt.data_mode, save=False)
    elif 'ChiSig' in opt.data:
        sigdataset_test = SigDataset(opt.data, train=False, image_size=opt.imageSize)
    
    test_loader = DataLoader(sigdataset_test, batch_size=opt.batchSize, shuffle=False)
    
    models_path = os.path.join("saved_models", opt.name)
    if opt.model_type == 'v1':
        model = ViT_for_OSV()
    elif opt.model_type == 'v2':
        model = ViT_for_OSV_v2()
    elif opt.model_type == 'v3':
        model = ViT_for_OSV_v3(opt)
    elif opt.model_type == 'emd':
        model = ViT_for_OSV_emd(opt)
    else:
        return NotImplementedError
    model.cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(os.path.join(models_path, opt.load)))
    
    #bce = ContrastiveLoss()
    if opt.loss == 'con':
        bce = ContrastiveLoss()
    elif opt.loss == 'bce':
        bce = torch.nn.BCELoss()
    else:
        return NotImplementedError
    validation_loss = val_acc = 0.0
    np_loss = np.zeros(shape=(0,))
    gt_loss = np.zeros(shape=(0,))
    data_df = pd.DataFrame(columns=['img_path', 'pos_output', 'neg_output'])
    # for X_val, Y_val, path in tqdm(test_loader):
    for X_val, Y_val in tqdm(test_loader):
        model.eval()
        # validate your model
        X_val = X_val.view(-1,2,opt.imageSize,opt.imageSize).cuda()
        Y_val = Y_val.view(-1,1).cuda()
        #print(X_val.shape)
        #plt.imsave('img_0.png', X_val[0,0].squeeze().detach().cpu().numpy())
        #plt.imsave('img_1.png', X_val[0,1].squeeze().detach().cpu().numpy())
        #plt.close()
        
        pred_val = model(X_val)
        #pred_val = model.forward_test(X_val)

        #_ = model.forward_test(X_val)

        if isinstance(pred_val, tuple):
            loss_0 = bce(pred_val[0], Y_val.float())
            loss_1 = bce(pred_val[1], Y_val.float())
            loss_val = loss_0 + loss_1
        else:
            loss_val = bce(pred_val, Y_val.float())

        #loss_val = bce(pred_val, Y_val.float())
        validation_loss += loss_val.data
        if isinstance(pred_val, tuple):
            val_acc += get_pct_accuracy(pred_val[0], Y_val)
        else:
            val_acc += get_pct_accuracy(pred_val, Y_val)
        #val_acc += get_pct_accuracy(pred_val, Y_val)

        if isinstance(pred_val, tuple):
            pred_val_ = pred_val[0] + pred_val[1]
            np_loss = np.append(np_loss, pred_val_.cpu().detach().numpy())
        else:
            np_loss = np.append(np_loss, pred_val.cpu().detach().numpy())
        gt_loss = np.append(gt_loss, Y_val.cpu().detach().numpy())
        #return
    
    validation_loss /= (float)(len(test_loader))
    val_acc /= (float)(len(test_loader))

    print("Validation: Acc={}%, Loss={}".format(val_acc, validation_loss))
    
    #np_loss = 1- np_loss
    #gt_loss = 1- gt_loss
    #print(np_loss, gt_loss)
    metrics, thresh_optimal = compute_accuracy_roc(np_loss, gt_loss, step=5e-5)
    data_df = pd.DataFrame({"dist": np_loss, "y_true": gt_loss})
    data_gb = data_df.groupby("y_true")
    pos_dist = data_gb.get_group(1)["dist"]
    neg_dist = data_gb.get_group(0)["dist"]
    plt.hist(np.array(pos_dist), 200, facecolor='g', alpha=0.3)
    plt.hist(np.array(neg_dist), 200, facecolor='r', alpha=0.3)
    plt.savefig(f"./density.png")
    plt.close()
    

    print("d optimal: {}".format(thresh_optimal))
    print("Metrics obtained: \n" + '-'*50)
    print(f"Acc: {metrics['best_acc'] * 100 :.4f} %")
    print(f"FAR: {metrics['best_far'] * 100 :.4f} %")
    print(f"FRR: {metrics['best_frr'] * 100 :.4f} %")
    print('-'*50)
    
    find_eer(metrics['far_arr'], metrics['frr_arr'], metrics['d_arr'], "BHSig")
    plot_roc(np.array(metrics['tpr_arr']), np.array(metrics['far_arr']), "BHSig")
    print("SCORE: {}".format(auc(np.array(metrics['far_arr']), np.array(metrics['tpr_arr']))))
    return True

def main() -> None:
    if not opt.test_only:
        #with torch.autograd.detect_anomaly():
        train(opt)
    else:
        test(opt)


if __name__ == "__main__":
    main()
