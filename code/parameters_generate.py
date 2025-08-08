import torch as torch

parameters={}

parameters['dwi_channel_num']=13
parameters['dce_channel_num']=6

parameters['dwi_tensordata']=r'/kaggle/input/breastcaner-subtypes/dwi_tensordata'
parameters['dce_tensordata']=r'/kaggle/input/breastcaner-subtypes/dce_tensordata'
parameters['labels_tensordata']=r'/kaggle/input/breastcaner-subtypes/labels_tensordata'

parameters['dwi_test_tensordata']=r'/kaggle/input/breastcaner-subtypes/dwi_test_tensordata'
parameters['dce_test_tensordata']=r'/kaggle/input/breastcaner-subtypes/dce_test_tensordata'
parameters['labels_test_tensordata']=r'/kaggle/input/breastcaner-subtypes/labels_test_tensordata/labels_test_tensordata.pth'

parameters['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parameters['num_epochs']=1500
parameters['finetune_num_epochs']=20
parameters['batch_size']=32
parameters['input_size']=32
parameters['segnum']=5
parameters['classnum']=4

torch.save(parameters,r"/kaggle/input/breastcaner-subtypes/parameters")

model_dict={}
torch.save(model_dict,r"/kaggle/input/breastcaner-subtypes/model_dict/model_dict.pth")
fusion_model_dict={}

torch.save(fusion_model_dict,r"/kaggle/input/breastcaner-subtypes/fusion_model_dict/fusion_model_dict.pth")
