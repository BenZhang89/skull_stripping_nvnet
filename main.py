'''
@Author: Zhou Kai
@GitHub: https://github.com/athon2
@Date: 2018-11-30 09:53:44
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from utils import Logger,load_old_model
from train import train_epoch
from validation import val_epoch
from nvnet import NvNet
from metrics import CombinedLoss, SoftDiceLoss
from dataset import BratsDataset
from data import write_data_to_file
import glob
import pickle
import random
random.seed(2020)

config = dict()
config["cuda_devices"] = True
#train together ben
# config["labels"] = (1, 2, 4)
# config["model_file"] = os.path.abspath("whole_label_dice.h5")

# train seperatley
config["labels"] = (4,) # change label to train
config["model_file"] = os.path.abspath("single_label_{}_dice.h5".format(config["labels"][0]))
config["initial_learning_rate"] = 1e-5 #Ben
config["batch_size"] = 1
config["validation_batch_size"] = 1
config["image_shape"] = (128, 128, 128)
# config["image_shape"] = (160,192,128)
config["activation"] = "relu"
config["normalizaiton"] = "group_normalization"
config["mode"] = "trilinear"
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
config["input_shape"] = tuple([config["batch_size"]] + [config["nb_channels"]] + list(config["image_shape"]))
config["loss_k1_weight"] = 0.1
config["loss_k2_weight"] = 0.1
config["random_offset"] = True # Boolean. Augments the data by randomly move an axis during generating a data
config["random_flip"] = True  # Boolean. Augments the data by randomly flipping an axis during generating a data
# config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
config["result_path"] = "./checkpoint_models/"
config["dicom_file"] = '/gpfs/scratch/bz957/brats/data/MICCAI/MICCAI_BraTS_2019_Data_Training_preprocessed/*/*'
config["data_file"] = os.path.abspath("isensee_mixed_brats_data.h5")
config["training_file"] = os.path.abspath("isensee_mixed_training_ids.pkl")
config["validation_file"] = os.path.abspath("isensee_mixed_validation_ids.pkl")
config["test_file"] = os.path.abspath("isensee_mixed_test_ids.pkl")
config["saved_model_file"] = None #ben
# config["saved_model_file"] = os.path.abspath(config["result_path"] +
#                                              config["model_file"].split("/")[-1].split(".h5")[0]
#                                              + '/epoch_96_val_loss_0.3027_acc_0.8036.pth')  #changing the best checkpoit accordingly
config["overwrite"] = False  # If True, will create new files. If False, will use previously written files.
config["L2_norm"] = 1e-5
config["patience"] = 2
# config["lr_decay"] = 0.9
config["lr_decay"] = 0.9
config["epochs"] = 300
config["checkpoint"] = True  # Boolean. If True, will save the best model as checkpoint.
config["label_containing"] = True  # Boolean. If True, will generate label with overlapping.
config["VAE_enable"] = True  # Boolean. If True, will enable the VAE module.

def fetch_training_data_files(tr_percentage = 0.8, val_percentage = 0.1, te_percentage = 0.1, return_subject_ids=False):
    training_data_files = list()
    subject_ids = list()
    for subject_dir in glob.glob(config["dicom_file"]):
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    #train, val, test split here
    if not os.path.exists(config["training_file"]) or not os.path.exists(config["validation_file"]) or not os.path.exists(config["test_file"]):
        data_ids = [i for i in range(len(subject_ids))]
        random.shuffle(data_ids)
        data_nums = len(data_ids)
        with open(config["training_file"], 'wb') as f_tr:
            pickle.dump(data_ids[:round(data_nums*tr_percentage)], f_tr)
        with open(config["validation_file"], 'wb') as f_v:
            pickle.dump(data_ids[round(data_nums*tr_percentage):round(data_nums*(tr_percentage+val_percentage))], f_v)
        with open(config["test_file"], 'wb') as f_te:
            pickle.dump(data_ids[round(data_nums*(1-te_percentage)):], f_te)

    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files

def main():
    # convert input images into an hdf5 file
    if config["overwrite"] or not os.path.exists(config["data_file"]):
        training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)
        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"],
                           subject_ids=subject_ids)

    # init or load model
    print("init model with input shape",config["input_shape"])
    model = NvNet(config=config)
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, 
                           lr=config["initial_learning_rate"],
                           weight_decay = config["L2_norm"])
    start_epoch = 1
    if config["VAE_enable"]:
        loss_function = CombinedLoss(k1=config["loss_k1_weight"], k2=config["loss_k2_weight"])
    else:
        loss_function = SoftDiceLoss()
    # data_generator
    print("data generating")
    training_data = BratsDataset(phase="train", config=config)
    valildation_data = BratsDataset(phase="validate", config=config)

    
    train_logger = Logger(model_name=config["model_file"],header=['epoch', 'loss', 'acc', 'lr'])

    if config["cuda_devices"] is not None:
        # model = nn.DataParallel(model)  # multi-gpu training
        model = model.cuda()
        loss_function = loss_function.cuda()
        
    if not config["overwrite"] and config["saved_model_file"] is not None:
        if not os.path.exists(config["saved_model_file"]):
            raise Exception("Invalid model path!")
        model, start_epoch, optimizer = load_old_model(model, optimizer, saved_model_path=config["saved_model_file"])    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config["lr_decay"],patience=config["patience"])
    
    print("training on label:{}".format(config["labels"]))
    max_val_acc = 0.
    for i in range(start_epoch,config["epochs"]):
        train_epoch(epoch=i, 
                    data_set=training_data, 
                    model=model,
                    criterion=loss_function, 
                    optimizer=optimizer, 
                    opt=config, 
                    logger=train_logger) 
        
        val_loss, val_acc = val_epoch(epoch=i, 
                  data_set=valildation_data, 
                  model=model, 
                  criterion=loss_function, 
                  opt=config,
                  optimizer=optimizer, 
                  logger=train_logger)
        scheduler.step(val_loss)
        if config["checkpoint"] and val_acc > max_val_acc:
            max_val_acc = val_acc
            save_dir = os.path.join(config["result_path"], config["model_file"].split("/")[-1].split(".h5")[0])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_states_path = os.path.join(save_dir,'epoch_{0}_val_loss_{1:.4f}_acc_{2:.4f}.pth'.format(i, val_loss, val_acc))
            states = {
                'epoch': i + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_states_path)
            save_model_path = os.path.join(save_dir, "best_model_file.pth")
            if os.path.exists(save_model_path):
                os.system("rm "+save_model_path)
            torch.save(model, save_model_path)
        
if __name__ == '__main__':
    main()
