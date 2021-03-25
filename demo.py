import pyrealsense2 as rs
import numpy as np
import cv2
from torchvision import datasets, transforms
import torch
from PIL import Image, ImageTk
import model_v6
import time
import tkinter
#import pred_result

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

input_size = 128

# Name_dict = {
#     0:'Yun',
#     1:'girl1',
#     2:'sun',
#     3:'girl2',
#     4:'Pan',
#     5:'Bird',
#     6:'girl3',
#     7:'Cheng',
#     8:'Jyun',
#     9:'lu',
#     10:'kai',
#     11:'Yes_mid',
#     12:'Right_stop',
#     13:'Multiply',
#     14:'One',
#     15:'UNKNOWN'
# }

Name_dict = {
    23:'Yun',
    25:'Cheng',
    26:'Jyun',
    27:'lu',
    28:'kai',
    29:'zong',
    30:'YU-TING',
    31:'Multiply',
    model_v6.UNKNOWN: 'UNKNOWN'
}

# pred_result_per_class = np.load('/media/rrrjjj/OS/Users/rayve/Desktop/temp/pred_per_class.npy', allow_pickle=True)
# rgb_out_3_per_class = np.load('/media/rrrjjj/OS/Users/rayve/Desktop/temp/rgb_out_3_temp.npy', allow_pickle=True)
# inner_prod_80_per_class = np.load('/media/rrrjjj/OS/Users/rayve/Desktop/temp/inner_prod_80_temp.npy', allow_pickle=True)
# depth_out_3_per_class = np.load('/media/rrrjjj/OS/Users/rayve/Desktop/temp/depth_out_3_temp.npy', allow_pickle=True)
# layer_cat_per_class = np.load('/media/rrrjjj/OS/Users/rayve/Desktop/temp/v2_simplified_grayRGB_134_faces_all_model_v6_cfgikm_Lidar_helo_s0/layer_cat.npy', allow_pickle=True)
layer_cat_per_class = np.load('C:\\Users\\rayve\\Desktop\\temp\\v2_simplified_grayRGB_134_faces_all_model_v6_cfgikm_Lidar_helo_s0\\layer_cat.npy', allow_pickle=True)

def argmin(lst):
	return min(range(len(lst)), key=lst.__getitem__)

def modify_infer(rgb_out_3, depth_out_3, cat_80, pred, infer, color_img_size):
    # threshold_innerProd = 88
    # #threshold_innerProduct_rgb_3 = 15
    # threshold_norm_80 = 15
    # #threshold_innerProduct_depth_3 = 25

    threshold_l1_loss_160 = 0.71
    threshold_l1_loss_160_to_80 = 0.75
    threshold_l1_loss_80_to_55 = 0.9
    threshold_l1_loss_55 = 0.88

    l1_loss = [99999] * model_v6.CLASSES

    for i in range(len(pred)):
        
        lossfunc = torch.nn.L1Loss().cuda()
        for j in range(model_v6.CLASSES):
            if(j in Name_dict.keys()):
                if(j != model_v6.UNKNOWN):
                    l1_loss[j] = lossfunc(cat_80, layer_cat_per_class[j])

        # innerProduct = torch.dot(pred[i], torch.tensor(pred_result_per_class[infer]).cuda().float())
        # #innerProduct_rgb_3 = torch.dot(rgb_out_3[i], torch.tensor(rgb_out_3_per_class[infer]).cuda())
        # #innerProduct_depth_3 = torch.dot(depth_out_3[i], torch.tensor(depth_out_3_per_class[infer]).cuda())
        # #innerProduct_80 = torch.dot(depth_out_3[i], torch.tensor(inner_prod_80_per_class[infer]).cuda())
        # norm_80 = torch.norm(cat_80)
        # #print(innerProduct)
        
        


        
        # #print("innerProd_rgb_3 = " + str(innerProduct_rgb_3))
        # print("innerProduct = " + str(innerProduct))
        # print("norm_80 = " + str(norm_80))
        # #print("innerProd_depth_3 = " + str(innerProduct_depth_3))
        # # if(innerProduct  < threshold_innerProd):
        # #     infer = model_v6.UNKNOWN

        # if(norm_80 > threshold_norm_80 or innerProduct < threshold_innerProd):
        #     infer = model_v6.UNKNOWN

    
    print("l1_loss = ", l1_loss[23:32])
    l1_loss_min = argmin(l1_loss)
    print("l1_loss_min = " + str(l1_loss_min) + ", value = " + str(l1_loss[l1_loss_min]))
    infer = l1_loss_min

    if color_img_size > 160:
        if(l1_loss[infer] > threshold_l1_loss_160):
            infer = model_v6.UNKNOWN
    elif color_img_size > 80:
        if(l1_loss[infer] > threshold_l1_loss_160_to_80):
            infer = model_v6.UNKNOWN
    elif color_img_size > 55:
        if(l1_loss[infer] > threshold_l1_loss_80_to_55):
            infer = model_v6.UNKNOWN
    else:
        if(l1_loss[infer] > threshold_l1_loss_55):
            infer = model_v6.UNKNOWN

    # try:
    #     print(str(infer) + " : " + Name_dict[infer])
    # except Exception as e:
    #     print(str(infer) + " : " + "NOT IN Name_dict")


    return infer


def preprocess_test(rgb_image, depth_image):

    depth_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.Grayscale(num_output_channels = 1),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    rgb_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    rgb_transforms_gray = transforms.Compose([
        transforms.Resize(input_size),
        transforms.Grayscale(num_output_channels = 1),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])


    rgb_data = rgb_transforms(rgb_image)
    rgb_data_gray = rgb_transforms_gray(rgb_image)
    depth_data = depth_transforms(depth_image)

    rgb_data = rgb_data.cuda()
    rgb_data_gray = rgb_data_gray.cuda()
    depth_data = depth_data.cuda()

    rgb_data = rgb_data.unsqueeze(0)
    rgb_data_gray = rgb_data_gray.unsqueeze(0)
    depth_data = depth_data.unsqueeze(0)

    return rgb_data, depth_data, rgb_data_gray



# Name_dict = {
#     0:'Yun',
#     1:'Cheng',
#     2:'Jyun',
#     3:'lu',
#     4:'kai',
#     5:'Yes_mid',
#     6:'Right_stop',
#     7:'Multiply',

# } 

# model = torch.load('/media/rrrjjj/OS/Users/rayve/Desktop/temp/v2_simplified_grayRGB_134_faces_all_model_v6_cfgikm_Lidar_helo_s0/8.pth')
model = torch.load('C:\\Users\\rayve\Desktop\\temp\\v2_simplified_grayRGB_134_faces_all_model_v6_cfgikm_Lidar_helo_s0\\8.pth')
model.eval()

counter = 0
counter_correct = 0
counter_frame = 0
start = True
first_enter_tkinter_window = True
try:
    root = tkinter.Tk()
    
    while True:
        # if(start):
        #     start_time = time.time()
        #     start = False
        

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)   
        depth_frame = aligned_frames.get_depth_frame() 
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
 
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
       
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        faces = face_cascade.detectMultiScale(color_image, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))

        for (x, y, w, h) in faces:
            if(start):
                start_time_final = time.time()
                start = False
            
            start_time = time.time()

            depth_img = depth_image[y:y +  h, x : x + w]
            depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
            color_img = color_image[y:y +  h, x : x + w]
            
            counter = counter + 1
            counter_frame = counter_frame + 1

            if(counter == 51):
                final_end_time = time.time()
                print("final time = ", final_end_time - start_time_final)
                #exit()

            cv2.imshow('RGB', color_img)

            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

            color_img = Image.fromarray(np.uint8(color_img))
            depth_img = Image.fromarray(np.uint8(depth_img))


            print("=========================================================================")
            print("color_img.size = ", color_img.size[0])
            color_img_size = color_img.size[0]

            

            color_img, depth_img ,color_img_gray = preprocess_test(color_img, depth_img)

            rgb_out_3, depth_out_3, cat_80, pred = model(color_img, depth_img, color_img_gray)

            

            #print(pred)
            infer = torch.argmax(pred, dim = 1)[0].item()

            print("original_infer = ", infer)

            infer = modify_infer(rgb_out_3, depth_out_3, cat_80, pred, infer, color_img_size)

            print("modified_infer = ", infer)
            end_time = time.time()
            print("inference time = " , end_time - start_time)
            print("=========================================================================")


            if(infer == 31):
                counter_correct = counter_correct + 1
            
            print("counter_correct = " + str(counter_correct))
            print("counter = " + str(counter))
            #print("accuracy = " + str(counter_correct / counter))

            try:
                text_name = Name_dict[infer] + ": " +str(np.round(depth_frame.get_distance(int(x+(1/2)*w), int(y+(1/2)*h)),3))+"m"
            except Exception as e:
                text_name = "NOT_IN_Name_dict"

            cv2.rectangle(color_image, (x, y-5), (x+w, y+h), (255, 0, 0), 2)
            
            color_image=cv2.putText(color_image,text_name,(x,y-5),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2,cv2.LINE_AA)
        ###############################################

        #images = np.hstack((color_image, depth_colormap))
        
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)

        

        im = Image.fromarray(color_image)
        colorImgTk = ImageTk.PhotoImage(image=im)
        tkinter.Label(root, image = colorImgTk).pack()
        #root.mainloop()
        
 
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        
        
        
 
 
finally:
    pipeline.stop()
