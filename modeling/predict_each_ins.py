import torch
import json
from PIL import Image
from model import Custom_AlexNet
from torchvision.transforms import Compose, ToTensor
from torch.nn.functional import softmax
import warnings
warnings.simplefilter("ignore", Warning)
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import confusion_matrix
from datetime import timedelta
from evaluation import sklearn_Compatible_preds_and_targets, accuracy_score


#DataLoader that returns the file path as well
class MyJP2Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        hmi = Image.open(img_path)

        if self.transform:
            image = self.transform(hmi)
            
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        
        return (image, y_label, img_path)

    def __len__(self):
        return len(self.annotations)
    

def predict(checkpoint, test_loader, desc ):
    test_target_list=[]
    test_prediction_list=[]
    test_path_list = []
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.eval()
    print('***********************', desc, '*************************')
    with torch.no_grad():
        for d, t, path in test_loader:
            # Get data to cuda if possible
            d = d.to(device=device)
            t = t.to(device=device)
    #         pa = path.to(device=device)
            test_target_list.append(t)
            test_path_list.append(list(path))
    #         print(list(path))
            # forward pass
            s = test_model(d)
            #print("scores", s)

            # validation batch loss and accuracy
    #         l = criterion(s, t)
            p = softmax(s,dim=1)
    #         print(p[:,1])
            test_prediction_list.append(p[:,1])
            # accumulating the val_loss and accuracy
    #         val_loss += l.item()
            #val_acc += acc.item()
            del d,t,s,p
    a, b, c = sklearn_Compatible_preds_and_targets(test_prediction_list, test_target_list, test_path_list)
    preds = [int(i >=0.5) for i in a]
    print(accuracy_score(preds, b))
    prob_list = pd.DataFrame(
    {'timestamp': c,
     'flare_prob': a,
     'target': b
    })

    print(prob_list['target'].value_counts())
    prob_list['timestamp'] = prob_list['timestamp'].apply(lambda row: row[35:-4])
    prob_list['timestamp'] = pd.to_datetime(prob_list['timestamp'], format='%Y.%m.%d_%H.%M.%S')
    return prob_list
        

    

# Device for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
# device = torch.device('cpu')
torch.backends.cudnn.benchmark = True
print(device)


# Load Data without any transformation as we now issue predictions
datapath = '/data/hmi_jpgs_512/'
partition1_path = '../data_labeling/data_labels/simplified_data_labels/Fold1_val.csv'
partition2_path = '../data_labeling/data_labels/simplified_data_labels/Fold2_val.csv'
partition3_path = '../data_labeling/data_labels/simplified_data_labels/Fold3_val.csv'
partition4_path = '../data_labeling/data_labels/simplified_data_labels/Fold4_val.csv'

transformations = Compose([
    ToTensor()
])

part1 = MyJP2Dataset(csv_file = partition1_path, 
                             root_dir = datapath,
                             transform = transformations)
part2 = MyJP2Dataset(csv_file = partition2_path, 
                             root_dir = datapath,
                             transform = transformations)
part3 = MyJP2Dataset(csv_file = partition3_path, 
                             root_dir = datapath,
                             transform = transformations)
part4 = MyJP2Dataset(csv_file = partition4_path, 
                             root_dir = datapath,
                             transform = transformations)

part1_loader = DataLoader(dataset=part1, batch_size=24, num_workers=4, shuffle=False)
part2_loader = DataLoader(dataset=part2, batch_size=24, num_workers=4, shuffle=False)
part3_loader = DataLoader(dataset=part3, batch_size=24, num_workers=4, shuffle=False)
part4_loader = DataLoader(dataset=part4, batch_size=24, num_workers=4, shuffle=False)
# CUDA_LAUNCH_BLOCKING=1

#Load trained models weight and map them all to cpu. since this is just for prediction, won't take longer time in cpu.
#Also, these trained models can be validated in systems without gpu as well.
model_PATH1 = 'trained_models/new-fold1.pth'
model_PATH2 = 'trained_models/new-fold2.pth'
model_PATH3 = 'trained_models/new-fold3.pth'
model_PATH4 = 'trained_models/new-fold4.pth'
weights1 = torch.load(model_PATH1, map_location=torch.device("cpu"))
weights2 = torch.load(model_PATH2, map_location=torch.device("cpu"))
weights3 = torch.load(model_PATH3, map_location=torch.device("cpu"))
weights4 = torch.load(model_PATH4, map_location=torch.device("cpu"))
test_model = Custom_AlexNet(train=False).to(device)

#Issuing predictions from the trained models
fold1 = predict(weights1, part1_loader, 'Fold-1 Results')
fold2 = predict(weights2, part2_loader, 'Fold-2 Results')
fold3 = predict(weights3, part3_loader, 'Fold-3 Results')
fold4 = predict(weights4, part4_loader, 'Fold-4 Results')

"""
Generated Results looks like this

*********************** Fold-1 Results *************************
TP:  1720 FP:  1943 TN:  10511 FN:  614
(0.5809181730499171, 0.47177415659692445)
0    12454
1     2334
Name: target, dtype: int64
*********************** Fold-2 Results *************************
TP:  1155 FP:  3083 TN:  10772 FN:  457
(0.49398229446599073, 0.28724094275141926)
0    13855
1     1612
Name: target, dtype: int64
*********************** Fold-3 Results *************************
TP:  1585 FP:  2668 TN:  11640 FN:  779
(0.4840046650744297, 0.3629519598840223)
0    14308
1     2364
Name: target, dtype: int64
*********************** Fold-4 Results *************************
TP:  1706 FP:  2241 TN:  11791 FN:  984
(0.47449435808963475, 0.39911957177843904)
0    14032
1     2690
Name: target, dtype: int64
"""

#Saving predictions issued for each instances
fold1.to_csv(r'trained_models/validation_preds/fold1_res.csv', index=False, header=True, columns=['timestamp', 'flare_prob', 'target'])
fold2.to_csv(r'trained_models/validation_preds/fold2_res.csv', index=False, header=True, columns=['timestamp', 'flare_prob', 'target'])
fold3.to_csv(r'trained_models/validation_preds/fold3_res.csv', index=False, header=True, columns=['timestamp', 'flare_prob', 'target'])
fold4.to_csv(r'trained_models/validation_preds/fold4_res.csv', index=False, header=True, columns=['timestamp', 'flare_prob', 'target'])