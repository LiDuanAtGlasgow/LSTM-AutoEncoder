import torch
import os
import torch.nn as nn
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
import pandas as pd
import statistics as statistics
from statistics import mean as mean_stat
from statistics import stdev as stdev
from Tool import SelfNoise
from Tool import pytorch_ssim as ssim
####################################
####Test Model (Prediction Test)####
####################################
def train(models,all_mario_test_prediction_loader,net,device,AutoEncoder_Type):
    first_error,second_error,third_error=[],[],[]
    first_error_average,second_error_average,third_error_average=[],[],[]
    total_fem,total_sem,total_tem=[],[],[]
    total_fes,total_ses,total_tes=[],[],[]
    first_error_ssim,second_error_ssim,third_error_ssim=[],[],[]
    first_error_average_ssim,second_error_average_ssim,third_error_average_ssim=[],[],[]
    total_fem_ssim,total_sem_ssim,total_tem_ssim=[],[],[]
    total_fes_ssim,total_ses_ssim,total_tes_ssim=[],[],[]
    first_error_standard_deviation,second_error_standard_deviation,third_error_standard_deviation=[],[],[]
    first_error_standard_deviation_ssim,second_error_standard_deviation_ssim,third_error_standard_deviation_ssim=[],[],[]
    selfnoise=SelfNoise.Gaussian_Nosie()
    train_start_time=time.time()
    save_test_window_slide_image='./data/Window_Sliding/'
    if not os.path.exists(save_test_window_slide_image):
       os.makedirs(save_test_window_slide_image)
    loss_fn_test= torch.nn.MSELoss(size_average=False)
    mean=[]
    standard_deviation=[]
    save_figure_window_slide='./data/Window_Sliding_figure/'
    net=net[0]
    inputs=[]
    ssim_input=ssim.SSIM()

    if not os.path.exists(save_figure_window_slide):
        os.makedirs(save_figure_window_slide)
    for n in range(len(all_mario_test_prediction_loader)):
        mario_test_prediction_loader=all_mario_test_prediction_loader[n]
        print(len(all_mario_test_prediction_loader))
        for z, input_ in enumerate (mario_test_prediction_loader):
            inputs=input_[0]
            input_noise=selfnoise.forward(inputs)
            inputs=Variable(inputs)
            inputs=inputs.to(device)
            input_noise=Variable(input_noise)
            input_noise=input_noise.to(device)
            for i in range(0,len(inputs)-8,1):
                mse_stat=[]
                ssim_stat=[]
                input_pred=input_noise[i:i+3]
                if AutoEncoder_Type==1:
                    input_pred=input_pred[:,0:1,:,:]
                if AutoEncoder_Type==2:
                    input_pred=input_pred
                for t in range(3):
                    target_exp=inputs[i+t+3:i+t+4]
                    target_exp=Variable(target_exp)
                    if AutoEncoder_Type==1:
                        target_exp=target_exp[:,0:1,:,:]
                    if AutoEncoder_Type==2:
                        target_exp=target_exp
                    target_exp=target_exp.to(device)
                    inputs_encor=net.encoder(input_pred)
                    inputs_lstm=models(inputs_encor,device)
                    inputs_out=inputs_lstm.view(-1,64,32,32)
                    inputs_decor=net.decoder(inputs_out)
                    loss_function=loss_fn_test(target_exp,inputs_decor)
                    ssim_out=-ssim_input(target_exp,inputs_decor)
                    ssim_value=-ssim_out.item()
                    mse_stat.append(loss_function.item())
                    ssim_stat.append(ssim_value)
                    if (i+1)%((len(inputs)-9)+1)==0:
                        print("[Dataset:%d/%d][Outer_Batch:%d/%d][Batch:%d/%d][Procedure:%d/%d][Duration:%f][Test Loss:%f][SSIM: %f]"
                            %(n+1,len(all_mario_test_prediction_loader),z+1,len(mario_test_prediction_loader),i+1,len(inputs),t+1,3,time.time()-train_start_time,mean_stat(mse_stat),mean_stat(ssim_stat)))
                        train_start_time=time.time()
                        cat=torch.cat([input_pred,target_exp,inputs_decor])
                        save_image(cat.cpu(),os.path.join(save_test_window_slide_image,"%d_%d%d.png"%(time.time(),i+1,t+1)))
                    input_pred=torch.cat([input_pred[1:3],inputs_decor])
                first_error.append(mse_stat[0])
                second_error.append(mse_stat[1])
                third_error.append(mse_stat[2])
                first_error_ssim.append(ssim_stat[0])
                second_error_ssim.append(ssim_stat[1])
                third_error_ssim.append(ssim_stat[2])
            first_error_average.append(mean_stat(first_error))
            second_error_average.append(mean_stat(second_error))
            third_error_average.append(mean_stat(third_error))
            first_error_standard_deviation.append(stdev(first_error))
            second_error_standard_deviation.append(stdev(second_error))
            third_error_standard_deviation.append(stdev(third_error))
            first_error_average_ssim.append(mean_stat(first_error_ssim))
            second_error_average_ssim.append(mean_stat(second_error_ssim))
            third_error_average_ssim.append(mean_stat(third_error_ssim))
            first_error_standard_deviation_ssim.append(stdev(first_error_ssim))
            second_error_standard_deviation_ssim.append(stdev(second_error_ssim))
            third_error_standard_deviation_ssim.append(stdev(third_error_ssim))
    total_fem=mean_stat(first_error_average)
    total_sem=mean_stat(second_error_average)
    total_tem=mean_stat(third_error_average)
    total_fes=stdev(first_error_standard_deviation)
    total_ses=stdev(second_error_standard_deviation)
    total_tes=stdev(third_error_standard_deviation)
    total_fem_ssim=mean_stat(first_error_average_ssim)
    total_sem_ssim=mean_stat(second_error_average_ssim)
    total_tem_ssim=mean_stat(third_error_average_ssim)
    total_fes_ssim=stdev(first_error_standard_deviation_ssim)
    total_ses_ssim=stdev(second_error_standard_deviation_ssim)
    total_tes_ssim=stdev(third_error_standard_deviation_ssim)
    mean=[total_fem,total_sem,total_tem]
    standard_deviation=[total_fes,total_ses,total_tes]
    data_matrix=[mean,standard_deviation]
    mean_ssim=[total_fem_ssim,total_sem_ssim,total_tem_ssim]
    standard_deviation_ssim=[total_fes_ssim,total_ses_ssim,total_tes_ssim]
    data_matrix_ssim=[mean_ssim,standard_deviation_ssim]
    data_title=['Mean','Standard Deviation']
    image_type=['RGB','Depth']
    coordination=['The First Prediction','The Second Prediction', 'The Third Prediction']
    colors=['blue','yellow']
    labels=['Mean (MSE)', 'Standard Deviation (MSE)']
    colors_ssim=['green','red']
    labels_ssim=['Mean (SSIM)', 'Standard Deviation (SSIM)']
    for s in range (len(data_matrix)):        
        subplot=plt.figure()
        subplot.add_subplot(111)
        sbplt1=plt.subplot()
        data=pd.DataFrame({'coordination':coordination,'window_test_mse': data_matrix[s],'window_test_ssim':data_matrix_ssim[s]})
        sbplt1.plot('coordination','window_test_mse',data=data,color=colors[s],label=labels[s])
        sbplt1.set_xlabel('Epoch')
        sbplt1.set_ylabel('Window Test MSE Loss')
        plt.legend(loc="upper right")
        plt.grid(True)
        sbplt2=sbplt1.twinx()
        sbplt2.plot('coordination','window_test_ssim',data=data,color=colors_ssim[s],label=labels_ssim[s])
        sbplt2.set_ylabel('Window Test SSIM Loss')
        plt.legend(loc="upper left")
        if AutoEncoder_Type==1:
            image_type_single=image_type[1]
        if AutoEncoder_Type==2:
            image_type_single=image_type[0]
        plt.title(image_type_single+' '+data_title[s]+' of Loss of of Window Sliding')
        plt.savefig(save_figure_window_slide+'/'+image_type_single+'_'+data_title[s]+'_'+'test_prediction_%f_.png'%(time.time()),dip=100)
        print ("window test mse is:",data_matrix[s])
        print ("window test ssim is:",data_matrix_ssim[s])
    print("Tranining is finishded!")