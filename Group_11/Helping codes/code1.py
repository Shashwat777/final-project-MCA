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
parser.add_argument('--batch_size', type=int, default=10)
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
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    args.train_files = "data/train.csv"
    args.test_files = pth
    args.val_files = "data/val.csv"
    # trainset = SMP_data(csv_file=args.train_files, root_dir=args.train_path, transform=transform)
    # valset = SMP_data(csv_file=args.val_files, root_dir=args.val_path, transform=transform)
    testset = SMP_data(csv_file=args.test_files, root_dir=args.test_path, transform=transform)
  
    

    # train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=True)
    # val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,drop_last=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # if args.train:
    #     loss_fn = nn.MSELoss()
    #     lr=args.lr
    #     weight_decay = 0.001
    #     optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    
    #     avg_train_loss_0 = 100
    #     for epoch in range(args.warm_start_epoch, args.epochs):
    #         batch_train_losses = []
    #         model.train()
    #         # training
    #         for num, data in enumerate(train_loader):

    #             img = data['img'].to(device)
    #             meta= data['meta'].to(device)
    #             tags = data['tags'] 
    #             title = data['title'] 
    #             category = data['category'] 
    #             text = [category,tags,title]
    #             label = data['label'].to(device)

    #             out = model(img, text, meta)       
                
    #             train_loss = loss_fn(out, label)
    #             batch_train_losses.append(train_loss.item())

    #             optimizer.zero_grad()
    #             train_loss.backward()
    #             nn.utils.clip_grad_value_(model.parameters(), 1.)
    #             nn.utils.clip_grad_norm_(model.parameters(), 10.)
    #             optimizer.step()

    #             if num % 50 == 0:
    #                 print('Epoch: %d/%d | Step: %d/%d | loss: %.4f' %(epoch + 1, args.epochs, num + 1, len(trainset) // args.batch_size + 1, train_loss.item()))

    #         # validation
    #         model.eval()
    #         avg_train_loss = round(sum(batch_train_losses) / len(batch_train_losses),5)
    #         print('Epoch %d averaged training loss: %.6f' % (epoch + 1, avg_train_loss))
    #         batch_train_losses = []

    #         batch_val_losses = []
    #         preds = []
    #         labels = []
    #         for num, data in enumerate(val_loader):
                
    #             img = data['img'].to(device)
    #             meta= data['meta'].to(device)
    #             tags = data['tags'] 
    #             title = data['title'] 
    #             category = data['category'] 
    #             text = [category,tags,title]
    #             label = data['label'].to(device)

    #             out = model(img, text, meta)   
                
    #             val_loss = loss_fn(out, label)
    #             batch_val_losses.append(val_loss.item())

    #             for i in range(out.shape[0]):
    #                 preds.append(out[i].item())
    #                 labels.append(label[i].item())

    #         lr = min(1e-5,lr * 0.9)
    #         optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
        
    #         mae=mean_absolute_error(labels, preds)
    #         mse=mean_squared_error(labels, preds)
    #         spearmanr_corr = stats.spearmanr(labels, preds)[0]
    #         print(mae)
    #         print(mse)
    #         print(spearmanr_corr)

    #         if avg_train_loss < avg_train_loss_0 - 0.001:
    #             avg_train_loss_0 = avg_train_loss
    #             torch.save(model, os.path.join(args.ckpt_path, 'epoch-%d-%.4f.pth' % (epoch+1,mae)))#.state_dict()
    #             print('Saved model. Testing...') #.state_dict()

     # testing
    
    
    print ("f")
    model.eval()
    with open('results.json','a+') as f:
        f.write('{"version": "VERSION 1.0","result": [')
        c=0
        
    
        for  num,data in enumerate(test_loader):
            print (str(c)+"s")
            if(c!=0):
                break
            c=1
            
            print (1)
       
            img = data['img'].to(device)
            print (2)
            meta= data['meta'].to(device)
            print (3)

            tags = data['tags'] 
            print (4)
            title = data['title'] 
            print (5)

            category = data['category'] 
            print (6)
            text = [category,tags,title]
            print (7)

            out = model(img, text, meta)  
            print (8)
            dic={}
        
            for i in range(out.shape[0]):
                result={}
                print (str(i)+"h")
                result['post_id'] = "post:"+ data['id'][i]
                result['popularity_score'] = round(out[i].item(),4)*(-100)
                dic[result['post_id']]=result['popularity_score'] 
               
                with open(pth.split("/")[1].split(".")[0]+'.json','a+') as f:
                    json.dump(result,f)
                    f.write(',')
            return (dic)
     

        with open('results.json','a+') as f:
            f.write('\b]}')

def random_text(n):
    l=[]
    for i in range (1, n):
        # number of words
       lene=random.randint(2,100)
       str_=""
       for word in range (1,lene):
            #   wordlength
              wrd=""
              wlen=random.randint(2,100)
              for alpha in range (1,wlen):
                  letter=chr(random.randint(97,121))
                  wrd=wrd+letter
              str_=wrd+" "+str_
       l.append(str_)
    return (l)
    
if __name__ == '__main__':

    args = parser.parse_args()
    model = SMP_model()
    
    # if args.warm_start_epoch:
    #     model = torch.load(os.path.join(args.ckpt_path, 'epoch-%d.pth' % args.warm_start_epoch))
       
    model=torch.load("ckpts/epoch-1-0.0569.pth",map_location=torch.device('cpu'))

    print('Loaded model')
    model = model.to(device)
    dicr=train(args, model,'data/test.csv')
    dicrkeys=dicr.keys()
    diclist=[]
    multi_sds=[]
    rndm_str=random_text(1000)
 
    

   

    for k in range (0, 5):
     l=[]
     for i in range (1,4):
        pth=""
        with open('data/test.csv', 'r') as file:
              reader = csv.reader(file)
              for row in reader:
                row[i]=rndm_str[random.randint(1,900)]
                pth="files/"+str(i)+"_"+str(k)+'file.csv'
                with open(pth, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row)
        
        # parser.add_argument('--test_files', type=str, default=pth)


        dic=train(args, model,pth)
        l.append(dic)
     diclist.append(l)
  


     sds=[]
     for j in diclist:
       l=[]
    
       for i in j:
           s=0
           for key in dicrkeys:
           
             score=i[key]
             scorer=dicr[key]
             diff=score-scorer
             if diff<0:
               diff=diff*(-1)
             s=s+diff
           s=s/len(dicrkeys)
           l.append(s)
       sds.append(l)
    
    
 
    arr=[]

    for i in range  (len(sds[0])):
        sum=0
        for j in range (len(sds)):
            sum=sum+sds[j][i]
        sum=sum/len(sds)
        arr.append(sum)
    print ("result")
    print (arr)



#     [[0.7837000000000001, 0.6819999999999996, 0.6821, 0.6265999999999999, 0.9227000000000001, 0.7755000000000006, 0.6976000000000002, 0.6040999999999999], [0.8362, 0.6796000000000002, 0.8104999999999998, 0.6478, 0.7595000000000001, 0.6652, 0.6775, 0.5980999999999999], [0.7680000000000002, 0.7363000000000001, 0.6621000000000001, 0.5306000000000001, 0.7951999999999999, 0.8201, 0.6858, 0.5720999999999998], [0.8513999999999998, 0.6572999999999998, 0.6338000000000003, 0.5110000000000002, 0.7215000000000003, 0.7528999999999999, 0.6949000000000001, 0.7135000000000001], [0.7097000000000004, 0.5318, 0.6322999999999998, 0.5198999999999998, 0.8077, 0.7441000000000001, 0.5941000000000003, 0.5778999999999999]]
# result
# [0.7898000000000002, 0.6573999999999999, 0.6841600000000001, 0.56718, 0.8013200000000001, 0.7515600000000001, 0.6699800000000001, 0.6131399999999999]