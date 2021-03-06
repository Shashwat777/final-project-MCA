# This code helps in oberving  the trend in the change of the popularity score with the mutation in clm 9 and 12 individually
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
                result['popularity_score'] = round(out[i].item(),4)
                dic[result['post_id']]=result['popularity_score'] *(-100)
               
                with open(pth.split("/")[1].split(".")[0]+'.json','a+') as f:
                    json.dump(result,f)
                    f.write(',')
            return (dic)
     
# gives average score for mutation
def get_av_score_mutation(datalist,l):
    returnee={}
    k=0
    for j in datalist:
        av=(sum(j.values()))/(len(j.values()))
        returnee[l[k]]=av
        k=k+1
    return (returnee)




    
if __name__ == '__main__':

    args = parser.parse_args()
    model = SMP_model()
    print ("loading model")
    
    model=torch.load("ckpts/epoch-1-0.0569.pth",map_location=torch.device('cpu'))
 

   
    model = model.to(device)
    print ("model loaded")
    # orig=train(args, model,'data/test.csv')
    # orig_keys=orig.keys()
    datalist1=[]
    datalist2=[]
    l=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
    posv=[]
    negv=[]
    for i in l:
        pthp="files/cln11pos"+"_"+str(i)+".csv"
        pthn="files/cln11neg"+"_"+str(i)+".csv"
       
        pos=[]
        neg=[]
        # in the below code result for other coloumns can be obtained  by changing row[8]->row[clm index]
        with open('data/test.csv', 'r') as file:
              reader = csv.reader(file)
           
              for row in reader:
                row[9]=str(float(row[9])*i)
                if(float(row[9])>0):
                    pos.append(float(row[9]))
                    with open(pthp, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row)
                    
                else:
                    neg.append(float(row[9]))
                  
                    with open(pthn, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row)
              v=sum(pos)/len(pos)
              posv.append(v)
              v=sum(neg)/len(neg)
              negv.append(v)
              

        

                
       
             
                        
         
                 
                
        mutated=train(args, model,pthp)
        
        datalist1.append(mutated)
        mutated=train(args, model,pthn)
        
        datalist2.append(mutated)
    
     #  prints the aveage score for  mutation on positive values
    print (get_av_score_mutation(datalist1,l))
    # prints the average score for mutation on  negative values
    print (get_av_score_mutation(datalist2,l))
    # Prints the average values of the coloumn after mutation on the positive values
    print ("positive averages")
    print (posv)
    # Prints the average values of the coloumn after mutation on the Negative values
    print ("negative  averages")
    print (negv)

#   result for batch of 10  for clm eight
# {0.5: 5.816000000000001, 0.6: 5.851000000000001, 0.7: 5.889000000000001, 0.8: 5.941000000000001, 0.9: 5.9940000000000015, 1: 6.0360000000000005, 1.1: 6.082, 1.2: 6.139, 1.3: 6.204, 1.5: 6.344}

# result for batch of 100 for clm eight
# res={0.0: 5.570499999999997, 0.1: 5.5799, 0.2: 5.590799999999998, 0.3: 5.6027, 0.4: 5.6188, 0.5: 5.636899999999998, 0.6: 5.655300000000002, 0.7: 5.6742, 0.8: 5.697200000000001, 0.9: 5.723000000000001, 1: 5.748900000000001, 1.1: 5.779600000000001, 1.2: 5.816499999999998, 1.3: 5.856899999999999, 1.5: 5.935699999999998, 1.6: 5.9763, 1.7: 6.0112999999999985, 1.8: 6.050099999999997, 1.9: 6.087499999999999, 2.0: 6.1221}
    
       
   
#   loading model
# model loaded clm 11
# {0.1: 5.608378378378378, 0.2: 5.592162162162162, 0.3: 5.576486486486487, 0.4: 5.553783783783785, 0.5: 5.534594594594595, 0.6: 5.51918918918919, 0.7: 5.498378378378378, 0.8: 5.477837837837837, 0.9: 5.458108108108108, 1: 5.448108108108109, 1.1: 5.446216216216215, 1.2: 5.447027027027027, 1.3: 5.454864864864864, 1.4: 5.464054054054051, 1.5: 5.4686486486486485, 1.6: 5.473783783783784, 1.7: 5.483513513513514, 1.8: 5.495945945945946, 1.9: 5.5148648648648635, 2: 5.537297297297298}
# {0.1: 5.974393939393941, 0.2: 5.991060606060605, 0.3: 5.986363636363638, 0.4: 5.982272727272729, 0.5: 5.980909090909092, 0.6: 5.9775757575757575, 0.7: 5.974848484848487, 0.8: 5.966212121212117, 0.9: 5.953181818181818, 1: 5.938636363636363, 1.1: 5.916363636363637, 1.2: 5.882121212121214, 1.3: 5.838030303030305, 1.4: 5.783787878787878, 1.5: 5.713333333333335, 1.6: 5.634090909090911, 1.7: 5.550909090909087, 1.8: 5.46878787878788, 1.9: 5.393484848484847, 2: 5.316212121212119}
# positive averages
# [0.08305033109729734, 0.16610066219459468, 0.24915099329189205, 0.33220132438918937, 0.4152516554864866, 0.4983019865837841, 0.5813523176810814, 0.6644026487783787, 0.7474529798756759, 0.8305033109729733, 0.9135536420702706, 0.9966039731675682, 1.0796543042648652, 1.1627046353621628, 1.24575496645946, 1.3288052975567575, 1.4118556286540542, 1.4949059597513519, 1.5779562908486493, 1.6610066219459465]
# negative  averages
# [-0.11836697685606058, -0.23673395371212116, -0.3551009305681817, -0.47346790742424233, -0.5918348842803032, -0.7102018611363634, -0.8285688379924251, -0.9469358148484847, -1.0653027917045448, -1.1836697685606064, -1.3020367454166668, -1.4204037222727268, -1.5387706991287873, -1.6571376759848502, -1.7755046528409077, -1.8938716296969693, -2.0122386065530313, -2.1306055834090896, -2.248972560265152, -2.3673395371212127]




# model loaded clm 8
# model loaded
# {0.1: 5.654242424242424, 0.2: 5.62090909090909, 0.3: 5.5915151515151535, 0.4: 5.563030303030302, 0.5: 5.535757575757574, 0.6: 5.5115151515151535, 0.7: 5.489393939393939, 0.8: 5.467575757575758, 0.9: 5.457878787878787, 1: 5.44969696969697, 1.1: 5.445454545454546, 1.2: 5.448787878787879, 1.3: 5.453030303030302, 1.4: 5.460000000000001, 1.5: 5.46939393939394, 1.6: 5.484545454545454, 1.7: 5.493636363636363, 1.8: 5.504545454545455, 1.9: 5.517575757575757, 2: 5.531212121212122}
# {0.1: 5.559999999999998, 0.2: 5.592285714285713, 0.3: 5.624142857142858, 0.4: 5.6612857142857145, 0.5: 5.7010000000000005, 0.6: 5.739714285714286, 0.7: 5.7785714285714285, 0.8: 5.823142857142859, 0.9: 5.866571428571427, 1: 5.909857142857143, 1.1: 5.957, 1.2: 6.010285714285715, 1.3: 6.066999999999998, 1.4: 6.12257142857143, 1.5: 6.175571428571425, 1.6: 6.228428571428571, 1.7: 6.276285714285714, 1.8: 6.3292857142857155, 1.9: 6.37885714285714, 2: 6.42385714285714}
# positive averages
# [0.06362999999999999, 0.12725999999999998, 0.19088999999999992, 0.25451999999999997, 0.31815000000000015, 0.38177999999999984, 0.44541000000000014, 0.5090399999999999, 0.5726700000000003, 0.6363000000000003, 0.6999300000000002, 0.7635599999999997, 0.8271900000000001, 0.8908200000000003, 0.9544499999999996, 1.0180799999999999, 1.0817100000000002, 1.1453400000000007, 1.2089699999999997, 1.2726000000000006]
# negative  averages
# [-0.06987299999999998, -0.13974599999999995, -0.20961900000000003, -0.2794919999999999, -0.3493650000000003, -0.41923800000000006, -0.48911099999999974, -0.5589839999999998, -0.628857, -0.6987300000000006, -0.7686029999999998, -0.8384760000000001, -0.908349000000001, -0.9782219999999995, -1.0480949999999996, -1.1179679999999996, -1.1878409999999997, -1.257714, -1.3275870000000012, -1.3974600000000013]





# clm 9


# {0.1: 5.679230769230769, 0.2: 5.651153846153846, 0.3: 5.63, 0.4: 5.612692307692309, 0.5: 5.599615384615385, 0.6: 5.600384615384616, 0.7: 5.607692307692308, 0.8: 5.617692307692308, 0.9: 5.629615384615386, 1: 5.640384615384615, 1.1: 5.656923076923078, 1.2: 5.677692307692307, 1.3: 5.6984615384615385, 1.4: 5.7226923076923075, 1.5: 5.7457692307692305, 1.6: 5.773076923076923, 1.7: 5.805769230769233, 1.8: 5.836923076923076, 1.9: 5.8684615384615375, 2: 5.895769230769232}
# {0.1: 5.916233766233765, 0.2: 5.903766233766234, 0.3: 5.8932467532467525, 0.4: 5.8810389610389615, 0.5: 5.866753246753246, 0.6: 5.8496103896103895, 0.7: 5.8342857142857145, 0.8: 5.8212987012987005, 0.9: 5.812207792207792, 1: 5.803636363636363, 1.1: 5.797012987012984, 1.2: 5.794675324675326, 1.3: 5.794155844155844, 1.4: 5.796623376623376, 1.5: 5.803116883116884, 1.6: 5.8114285714285705, 1.7: 5.82220779220779, 1.8: 5.834025974025977, 1.9: 5.843636363636364, 2: 5.854285714285712}
# positive averages
# [0.07456076923076924, 0.14912153846153847, 0.22368230769230776, 0.29824307692307694, 0.37280384615384593, 0.4473646153846155, 0.5219253846153846, 0.5964861538461539, 0.6710469230769232, 0.7456076923076919, 0.8201684615384617, 0.894729230769231, 0.9692900000000002, 1.0438507692307692, 1.1184115384615385, 1.1929723076923078, 1.2675330769230764, 1.3420938461538463, 1.4166546153846151, 1.4912153846153837]
# negative  averages
# [-0.06122402597402601, -0.12244805194805201, -0.183672077922078, -0.24489610389610403, -0.30612012987012993, -0.367344155844156, -0.4285681818181819, -0.48979220779220806, -0.5510162337662341, -0.6122402597402599, -0.6734642857142858, -0.734688311688312, -0.7959123376623376, -0.8571363636363638, -0.91836038961039, -0.9795844155844161, -1.0408084415584418, -1.1020324675324682, -1.1632564935064944, -1.2244805194805197]
