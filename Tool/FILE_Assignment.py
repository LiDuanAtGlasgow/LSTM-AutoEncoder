import cv2 as cv
import os
from openpyxl import load_workbook
import time

location="./File_Detail.xlsx"
workbook=load_workbook(location)
sheet=workbook.active

file_sheet=sheet["K5:K6"]
prefix_sheet=sheet["H14:H16"]
volume_sheet=sheet["I14:I16"]
x1_sheet=sheet["B121:B150"]
x2_sheet=sheet["C121:C150"]
move="_move"
pant="pant"
afflix=".png"

file_name=[]
prefix_name=[]
volume_name=[]
x1_name=[]
x2_name=[]

for t in range (len(file_sheet)):
    file_name.append(file_sheet[t][0].value)
for t in range (len(prefix_sheet)):
    prefix_name.append(prefix_sheet[t][0].value)
for t in range (len(volume_sheet)):
    volume_name.append(volume_sheet[t][0].value)
for t in range (len(x1_sheet)):
    x1_name.append(x1_sheet[t][0].value)
    x2_name.append(x2_sheet[t][0].value)

folder_location='./clothes/'
target_location="./mask_clothes/"
if not os.path.exists(target_location):
    os.mkdir(target_location)

account_move=0
page=0
for i in range (len(prefix_name)):
    prefix=prefix_name[i]
    page+=1
    if page>3:
        page=1
    file_location=folder_location+prefix+"/"+prefix+str(page)
    save_location_depth=target_location+prefix+str(page)+"_depth"+"/"
    if not os.path.exists(save_location_depth):
        os.mkdir(save_location_depth)
    save_location_rgb=target_location+prefix+str(page)+"_rgb"+"/"
    if not os.path.exists(save_location_rgb):
        os.mkdir(save_location_rgb)
    move=volume_name[i]
    x1_sec=x1_name[account_move:account_move+move]
    x2_sec=x2_name[account_move:account_move+move]
    account_move+=move
    for idx in range (len(x1_sec)):
            image_location=file_location+"_"+"move"+str(idx+1)+"/"
            x1=x1_sec[idx]
            x2=x2_sec[idx]
            if 0<x1%20<10:
                x1=x1-(x1%20)
            start_time=time.time()
            for x1_idx in range (x1):
                image=image_location+"pant1_x1img"+str(x1_idx+1)+"_"
                depth_location=image+"depth"+afflix
                rgb_location=image+"rgb"+afflix
                mask_location=image+"mask"+afflix
                print("the value of rgb_location is:",rgb_location)
                depth_image=cv.imread(depth_location)
                rgb_image=cv.imread(rgb_location)
                mask_image=cv.imread(mask_location)
                mask_not=cv.bitwise_not(mask_image)
                depth_mask=cv.bitwise_or(depth_image,mask_not)
                dim=(640,480)
                rgb_crop=cv.resize(rgb_image,dim,interpolation=cv.INTER_AREA)
                rgb_mask=cv.bitwise_or(rgb_crop,mask_not)
                if 0<=x1_idx<=8:
                    depth_target=target_location+prefix+str(page)+"_depth"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x1_img"+"00"+str(x1_idx+1)+afflix
                    rgb_target=target_location+prefix+str(page)+"_rgb"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x1_img"+"00"+str(x1_idx+1)+afflix
                if 9<=x1_idx<=98:
                    depth_target=target_location+prefix+str(page)+"_depth"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x1_img"+"0"+str(x1_idx+1)+afflix
                    rgb_target=target_location+prefix+str(page)+"_rgb"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x1_img"+"0"+str(x1_idx+1)+afflix
                if 99<=x1_idx:
                    depth_target=target_location+prefix+str(page)+"_depth"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x1_img"+str(x1_idx+1)+afflix
                    rgb_target=target_location+prefix+str(page)+"_rgb"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x1_img"+str(x1_idx+1)+afflix
                
                cv.imwrite(depth_target,depth_mask)
                cv.imwrite(rgb_target,rgb_mask)
            if not x1%20==0:
                add_index=20-x1%20
                for add_idx in range(add_index):
                    if 9<=x1<=98:
                        depth_duplicate=cv.imread(target_location+prefix+str(page)+"_depth"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x1_img"+"0"+str(x1)+afflix)
                        rgb_duplicate=cv.imread(target_location+prefix+str(page)+"_rgb"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x1_img"+"0"+str(x1)+afflix)
                    if 99<=x1:
                        depth_duplicate=cv.imread(target_location+prefix+str(page)+"_depth"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x1_img"+str(x1)+afflix)
                        rgb_duplicate=cv.imread(target_location+prefix+str(page)+"_rgb"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x1_img"+str(x1)+afflix)
                    if 9<=x1+add_idx<=98:
                       target_location_depth=target_location+prefix+str(page)+"_depth"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x1_img"+"0"+str(x1+add_idx+1)+afflix
                       target_location_rgb=target_location+prefix+str(page)+"_rgb"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x1_img"+"0"+str(x1+add_idx+1)+afflix
                    if 99<=x1+add_idx:
                       target_location_depth=target_location+prefix+str(page)+"_depth"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x1_img"+str(x1+add_idx+1)+afflix
                       target_location_rgb=target_location+prefix+str(page)+"_rgb"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x1_img"+str(x1+add_idx+1)+afflix
                    cv.imwrite(target_location_depth,depth_duplicate)
                    cv.imwrite(target_location_rgb,rgb_duplicate)
            print (prefix+str(page),"_move",str(idx+1),"_pant1_x1img","is finsihed"," ","Duration is %f"%(time.time()-start_time))
            
            print("The Original Value of x2 is:",x2)
            if 0<x2%20<10:
                x2=x2-(x2%20)
            print ("The Processed Value of x2 is:",x2)
            start_time=time.time()
            for x2_idx in range (x2):
                image=image_location+"pant1_x2img"+str(x2_idx+1)+"_"
                depth_location=image+"depth"+afflix
                rgb_location=image+"rgb"+afflix
                mask_location=image+"mask"+afflix
                depth_image=cv.imread(depth_location)
                rgb_image=cv.imread(rgb_location)
                mask_image=cv.imread(mask_location)
                mask_not=cv.bitwise_not(mask_image)
                dim=(640,480)
                rgb_crop=cv.resize(rgb_image,dim,interpolation=cv.INTER_AREA)
                depth_mask=cv.bitwise_or(depth_image,mask_not)
                rgb_mask=cv.bitwise_or(rgb_crop,mask_not)
                if 0<=x2_idx<=8:
                    depth_target=target_location+prefix+str(page)+"_depth"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x2_img"+"00"+str(x2_idx+1)+afflix
                    rgb_target=target_location+prefix+str(page)+"_rgb"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x2_img"+"00"+str(x2_idx+1)+afflix
                if 9<=x2_idx<=98:
                    depth_target=target_location+prefix+str(page)+"_depth"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x2_img"+"0"+str(x2_idx+1)+afflix
                    rgb_target=target_location+prefix+str(page)+"_rgb"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x2_img"+"0"+str(x2_idx+1)+afflix
                if 99<=x2_idx:
                    depth_target=target_location+prefix+str(page)+"_depth"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x2_img"+str(x2_idx+1)+afflix
                    rgb_target=target_location+prefix+str(page)+"_rgb"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x2_img"+str(x2_idx+1)+afflix
                print("the value of rgb_target is:",rgb_target)
                cv.imwrite(depth_target,depth_mask)
                cv.imwrite(rgb_target,rgb_mask)
            if not x2%20==0:
                print ("The original value of x2 is:",x2)
                add_index=20-x2%20
                for add_idx in range(add_index):
                    if 9<=x2<=98:
                        depth_duplicate=cv.imread(target_location+prefix+str(page)+"_depth"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x2_img"+"0"+str(x2)+afflix)
                        rgb_duplicate=cv.imread(target_location+prefix+str(page)+"_rgb"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x2_img"+"0"+str(x2)+afflix)
                    if 99<=x2:
                        depth_duplicate=cv.imread(target_location+prefix+str(page)+"_depth"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x2_img"+str(x2)+afflix)
                        rgb_duplicate=cv.imread(target_location+prefix+str(page)+"_rgb"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x2_img"+str(x2)+afflix)
                    if 9<=x2+add_idx<=98:
                       target_location_depth=target_location+prefix+str(page)+"_depth"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x2_img"+"0"+str(x2+add_idx+1)+afflix
                       target_location_rgb=target_location+prefix+str(page)+"_rgb"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x2_img"+"0"+str(x2+add_idx+1)+afflix
                    if 99<=x2+add_idx:
                       target_location_depth=target_location+prefix+str(page)+"_depth"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x2_img"+str(x2+add_idx+1)+afflix
                       target_location_rgb=target_location+prefix+str(page)+"_rgb"+"/"+prefix+str(page)+"_move"+str(idx+1)+"_pant1_x2_img"+str(x2+add_idx+1)+afflix
                    print ("The current value of x2 is:",x2+add_idx+1)
                    cv.imwrite(target_location_depth,depth_duplicate)
                    cv.imwrite(target_location_rgb,rgb_duplicate)
            print (prefix+str(page),"_move",str(idx+1),"_pant1_x2img","is finsihed"," ","Duration is :%f"%(time.time()-start_time))                
print ("The mask is finished!")

