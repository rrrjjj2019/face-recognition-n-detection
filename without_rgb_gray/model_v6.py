import torch
import torch.nn as nn
import torch.nn.functional as F
import flatten

CLASSES = 135
UNKNOWN = 135 - 1
FEATURE_CLASSES_OFFSET_DOWN = 18
FEATURE_CLASSES_OFFSET_UP = 33

class ecnn_fit(nn.Module):  
    def __init__(self):
        super(ecnn_fit, self).__init__()
        bn = True
        
        # ====================rgb 3=====================
        self.rgb_feature3_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.rgb_feature3_2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.rgb_feature3_3 = nn.Sequential(
            nn.MaxPool2d(2),
            #nn.Flatten(),
            flatten.Flatten(),
            nn.Linear(32768, 1000), #original 32*16*16
            nn.BatchNorm1d(40, affine = False),
            nn.Dropout(p=0.6),
        )

        # ====================rgb 3 gray=====================
        self.rgb_feature3_1_gray = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.rgb_feature3_2_gray = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.rgb_feature3_3_gray = nn.Sequential(
            nn.MaxPool2d(2),
            #nn.Flatten(),
            flatten.Flatten(),
            nn.Linear(32768, 350), #original 32*16*16
            nn.BatchNorm1d(40, affine = False),
            nn.Dropout(p=0.6),
        )

        
        # ====================depth 3=====================
        self.depth_feature3_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.depth_feature3_2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.depth_feature3_3 = nn.Sequential(
            nn.MaxPool2d(2),
            #nn.Flatten(),
            flatten.Flatten(),
            nn.Linear(32 * 16 * 16 * 4, 1000),
            nn.BatchNorm1d(40, affine = False),
            nn.Dropout(p=0.6)
        )

        

        self.classifier_concat = nn.Sequential(
            nn.Linear(80, CLASSES), #16 class
        )
            

    def forward(self,x , y):
        x1 = self.rgb_feature3_1(x)
        x1 = self.rgb_feature3_2(x1)
        x1 = self.rgb_feature3_3(x1)

        # x1_gray = self.rgb_feature3_1_gray(x_gray)
        # x1_gray = self.rgb_feature3_2_gray(x1_gray)
        # x1_gray = self.rgb_feature3_3_gray(x1_gray)

        y1 = self.depth_feature3_1(y)
        y1 = self.depth_feature3_2(y1)
        y1 = self.depth_feature3_3(y1)

        cat = torch.cat((x1, y1),1)
        
        cat_80 = cat.view(cat.size(0), -1)

        cat = self.classifier_concat(cat_80)
        
        return x1, y1, cat_80, cat
