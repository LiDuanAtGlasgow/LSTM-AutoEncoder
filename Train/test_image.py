import torch
import os
import time
from torch.autograd import Variable
from torch.utils import save_image
import pandas as pd
import matplotlib.pyplot as plot

def test_image(model_rgb,model_depth,batch_size,test_rgb_loader,test_depth_loader,device):
    save_dir='./data/test_image'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_test='./data/test_loss'
    All_item_time=[int(len(test_rgb_loader)/10),int(len(test_depth_loader))]
    start_time=time.time()
    eachtime_loss=0.00
    test_data={[],[]}
    batch=[]
    All_loader=[test_rgb_loader,test_depth_loader]
    All_model=[model_rgb,model_depth]
    All_color={'red','blue'}
    All_title={'Auto Encoder Loss for RGB Image','Auto Encoder Loss for Depth Image'}
    for i in range (0,len(model)):
        test_loader=All_model[i]
        model=All_model[i]
        tes_eps_los=test_data[i]
        item_time=All_item_time[i]
        color=All_color[i]
        title=All_title[i]   
        for t,input_ in enumerate(test_loader,0):
            inputs=input_[0].to(device)
            inputs=Variable(inputs)
            inputs_rev=model[inputs]
            loss=model.loss_function(inputs_rev,inputs)
            eachtime_loss+=loss.item()
            if ((t+1)%(item_time+1)==1):
                print("Batch [%d/%d],Duration[%f],Avrage Loss:%f"
                %(t+1,len(test_loader),time.time()-start_time,eachtime_loss/item_time))
                n=min(inputs[0],8)
                cat=torch.cat(input[:n],inputs_rev[:n])
                save_image(cat.cpu(),os.path.join(save_dir,'/%f.png'%time.time()))
                average_loss_test=eachtime_loss/item_time
                tes_eps_los.append(average_loss_test)
                eachtime_loss=0.00
    
        df=pd.DataFrame({'x':range (0,len(All_loader[i]))'test':tes_eps_los})
        asplot=plot.figure()
        asplot.add_subplot(111)
        plot.plot('x','test', data=df,color=color,label='test_of_rgb')
        plot.xlabel('Epoch')
        plot.ylabel('AutoEncoder Test Loss')
        plot.title(title)
        plot.grid(True)
        plot.savefig(os.path.join(save_test,'/Test_Loss_AutoEncoder%d.png'%i),dip=100)
    print("The Training is Finished!")


        