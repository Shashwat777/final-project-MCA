# The following code provides the recommendations for the increase of  popularity score on the basis of analysis of a pretrained file
# -*- coding: utf-8 
import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import mean_squared_error,mean_absolute_error
import json
from smp_model import SMP_model
from smp_data import SMP_data
import csv
import random







os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cpu")

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--warm_start_epoch', type=int, default=1)


parser.add_argument('--test_path', type=str, default='images/')
parser.add_argument('--train_files', type=str, default='data/train.csv')
parser.add_argument('--val_files', type=str, default='data/val.csv')
# parser.add_argument('--test_files', type=str, default='data/test.csv')
parser.add_argument('--ckpt_path', type=str, default='ckpts')
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--test', type=bool, default=True)


def train(args, model,pth):
    
    transform = transforms.Compose(
        [
            transforms.Resize([224,224]),
          
            transforms.ToTensor(),
        ]
    )
    args.train_files = "data/train.csv"
    args.test_files = pth
    args.val_files = "data/val.csv"

    testset = SMP_data(csv_file=args.test_files, root_dir=args.test_path, transform=transform)
  
    

    
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    
    
 
    model.eval()
    with open('results.json','a+') as f:
        f.write('{"version": "VERSION 1.0","result": [')
        c=0
        
    
        for  num,data in enumerate(test_loader):
            
            if(c!=0):
                break
            c=1
            
      
       
            img = data['img'].to(device)
      
            meta= data['meta'].to(device)


            tags = data['tags'] 

            title = data['title'] 
      

            category = data['category'] 
  
            text = [category,tags,title]
     

            out = model(img, text, meta)  
            dic={}
        
            for i in range(out.shape[0]):
                result={}
        
                result['post_id'] = "post:"+ data['id'][i]
                result['popularity_score'] = round(out[i].item(),4)*(-100)
                dic[result['post_id']]=result['popularity_score'] 
               
                with open("scores/"+pth.split("/")[1].split(".")[0]+'.json','a+') as f:
                    json.dump(result,f)
                    f.write(',')
            return (dic)
def analyze(mlist):
    rlist=[]
    for elem in mlist:
        mdict={}
        mdict["file:"]=elem[0]
        v12=float(elem[11])
        v9=float(elem[8])
        if(v12>0):
            if(v12>0 and v12<0.9966039731675682):
                mdict["is"]=-1
            else:
                mdict["is"]=1
        else:
            if(v12<-0.23673395371212116):
                mdict["is"]=1
            else:
                mdict["is"]=-1
        if(v9>0):
            if(v9>0 and v9<0.7):
                mdict["ui"]=-1
            else:
                mdict["ui"]=1
        else:
            mdict["ui"]=-1
        rlist.append(mdict)
    return(rlist)

        











def give_recomendation(pth,c):
     mlist=[]
     with open(pth, 'r') as file:
              reader = csv.reader(file)
              for row in reader:
                  mlist.append(row)
     res=analyze(mlist)

     file=open("recommendations/"+str(c)+".txt", 'a+')
     count=0
     for j in res:
             file.write(str(count)+")   ")
             name=j["file:"]
             r12=j["is"]
             r9=j["ui"]
             if(r12==1):
                 if(r9==1):
                     file.write("file:"+name+"  Recommendation: Increase the value of col 12 and 9")
                 else:
                     file.write("file:"+name+"  Recommendation: Increase the value of col 12 and  decrease the values of col 9")
             else:
                 if(r9==1):
                      file.write("file:"+name+"  Recommendation: Decrease the value of col 12 and increase the value of col 9")
                 else:
                      file.write("file:"+name+"  Recommendation: Decrease the value of coloumn 12 and 9")
             file.write('\n \n \n')
             count=count+1
     file.close()
def optimise_file(loc1,per,loc2):
     with open(loc1, 'r') as file:
              reader = csv.reader(file)
              for row in reader:
                v12=float(row[11])
                row[11]=float(row[11])

                v9=float(row[8])
                row[8]=float(row[8])
                if(v12>0):
                    if(v12>0 and v12<0.9966039731675682):
               
                            row[11]=row[11]*(1-(per/100))
                    else:
                            row[11]=row[11]*(1+(per/100))
                else:
                    if(v12<-0.23673395371212116):
                                 row[11]=row[11]*(1-(per/100))
                    else:
                                 row[11]=row[11]*(1+(per/100))
     
                if(v9>0):
                        if(v9>0 and v9<0.7):
                            row[8]=row[8]*(1-(per/100))
                        else:
                            row[8]=row[8]*(1+(per/100))
                else:
                       row[8]=row[8]*(1+(per/100))
                row[9]=float(row[9])
                if(float(row[9])<-0.7499):
                     row[9]=row[9]*(1+(per/100))
                elif(float(row[9])>-0.7499 and float(row[9]) <0):
                    row[9]=row[9]*(1-(per/100))
                elif(float(row[9])<0.380299 and float(row[9]) >0):
                    row[9]=row[9]*(1-(per/100))
                else:
                    row[9]=row[9]*(1-(per/100))
            




                

                with open(loc2, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row)
    
         


              
          


    





def split_test(pth,times,length):
     l=[]
     with open(pth, 'r') as file:
              reader = csv.reader(file)
              c=1
              j=500//times
              for row in reader:
                k=c//j
                
        
                pth="files/"+"test_"+str(k)+'.csv'
                if(pth not in l):
                    l.append(pth)

                with open(pth, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row)
                c=c+1
     return (l)


    
if __name__ == '__main__':

    args = parser.parse_args()
    model = SMP_model()
    print ("loading model")
    
    model=torch.load("ckpts/epoch-1-0.0569.pth",map_location=torch.device('cpu'))
 

   
    model = model.to(device)
    print ("model loaded")
    path_list='data/test.csv'
 
    c=0
    percent=300
    returnlist=[]
    
    
    loc="data/test.csv"

    original=train(args, model,loc)
   
   
    give_recomendation(loc,c)
    optimised_location= "files/"+str(c)+"optimised_test.csv"
    optimise_file(loc,percent,optimised_location)
    optimise=train(args, model,optimised_location)
    count=0
    diff=0
    for i in original.keys():
                      val_org=float(original[i])
                      val_opt=float(optimise[i])
                      prntg=(val_opt-val_org)/val_org
                      print (val_org,val_opt,prntg)
                      if(val_org<val_opt):
                             count=count+1
                             diff=diff+(val_opt-val_org)/val_org
    print ("Accuracy")
       
    print (count/5)






