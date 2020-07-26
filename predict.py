'''
@Author: Zhou Kai
@GitHub: https://github.com/athon2
@Date: 2018-11-30 09:53:44
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import nibabel as nib
import tables
import torch
from tqdm import tqdm
from dataset import BratsDataset
from nvnet import NvNet
from main import config
from utils import pickle_load

config["best_model_file"] = os.path.join(config["result_path"], config["model_file"].split("/")[-1].split(".h5")[0], "best_model_file.pth") # Load the best model
config["checkpoint_file"] = None  # Load model from checkpoint file
config["pred_data_file"] = os.path.abspath("isensee_mixed_brats_data.h5") # data file for prediction
config["prediction_dir"] = os.path.abspath("./prediction/")
config["load_from_data_parallel"] = False # Load model trained on multi-gpu to predict on single gpu.
        
def init_model_from_states(config):
    print("Init model...")
    model = NvNet(config=config)

    if config["cuda_devices"] is not None:
        # model = torch.nn.DataParallel(model)   # multi-gpu training
        model = model.cuda()
    checkpoint = torch.load(config["best_model_file"])  
    state_dict = checkpoint["state_dict"]
    if not config["load_from_data_parallel"]:
        model.load_state_dict(state_dict)
    else:
        from collections import OrderedDict     # Load state_dict from checkpoint model trained by multi-gpu 
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)

    return model


def predict(model, model_prediction_dir):
    print("Predicting...")
    model.eval()
    try:
        data_file = tables.open_file(config["pred_data_file"], "r")
        #load test idx
        test_idxs = pickle_load(config["test_file"])
        # for index in range(len(data_file.root.data)):
        for index in test_idxs:
            if "subject_ids" in data_file.root:
                case_dir = os.path.join(model_prediction_dir, data_file.root.subject_ids[index].decode('utf-8'))
            else:
                case_dir = os.path.join(output_dir, "pred_case_{}".format(index))
            if not os.path.exists(case_dir):
                os.makedirs(case_dir)
                
            data_array = np.asarray(data_file.root.data[index])[np.newaxis]
            affine = data_file.root.affine[index]
            
            assert data_array.shape == config["input_shape"], "Wrong data shape!Expected {0}, but got {1}.".format(config["input_shape"], data_array.shape)
            if config["cuda_devices"] is not None:
                inputs = torch.from_numpy(data_array)
                inputs = inputs.type(torch.FloatTensor)
                inputs = inputs.cuda()
            with torch.no_grad():
                if config["VAE_enable"]:
                    outputs, distr = model(inputs)
                else:
                    outputs = model(inputs)   
            output_array = np.asarray(outputs.tolist())
            output_array = output_array > 0.5
            print(output_array[0][0].shape)
            #for whole labels ben
            # output_image_whole = nib.Nifti1Image(output_array[0][0].astype('int'), affine)
            # output_image_enhancing = nib.Nifti1Image(output_array[0][1].astype('int'), affine)
            # output_image_core = nib.Nifti1Image(output_array[0][2].astype('int'), affine)
            # output_image_whole.to_filename(os.path.join(case_dir, "prediction_label_core.nii.gz"))
            # output_image_enhancing.to_filename(os.path.join(case_dir, "prediction_label_edema.nii.gz"))
            # output_image_core.to_filename(os.path.join(case_dir, "prediction_label_enhancing.nii.gz"))
            #for single label
            output_image = nib.Nifti1Image(output_array[0][0].astype('int'), affine)
            output_image.to_filename(os.path.join(case_dir, "prediction_label_{}.nii.gz".format(str(config["labels"][0]))))

    finally:
        data_file.close()
    
    
if __name__ == "__main__":
    if config["checkpoint_file"] is not None:
        model = init_model_from_states(config)
    else:
        model = torch.load(config["best_model_file"])
    model_prediction_dir = os.path.join(config["prediction_dir"],os.path.basename(config["model_file"].split(".h5")[0]))
    
    predict(model, model_prediction_dir)