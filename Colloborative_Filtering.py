import pandas as pd
import math
import gzip
import tarfile
import numpy as np

def getFiles(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def buildDataFrame(path):
  i = 0
  df = {}
  for d in getFiles(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = buildDataFrame('C:\\Users\\SarathKumar\\Desktop\\Project\\reviews_Amazon_Instant_Video_5.json.gz')
c=df.groupby('reviewerID')['asin'].nunique()
d=c.add_suffix('').reset_index()
Avg_count_df = d.rename(columns={'asin': 'asin_Count'})

def avg_vot(ids):
    votes = sum(list(df.overall[df.reviewerID == ids]))/Avg_count_df.asin_Count[Avg_count_df.reviewerID == ids]
    return votes.values[0]


gr = df[['reviewerID','overall','asin']]


def cor_list(user1,user2):
    q = list(gr.asin[gr.reviewerID == user1])
    w = list(gr.asin[gr.reviewerID == user2])
    dict_item_vote={}
    intrq = list(set(q).intersection(w))
    for item in intrq:
        list_votes=[]
        list_votes.append(gr.overall[(gr.asin == item) & (gr.reviewerID == user1)].values[0])
        list_votes.append(gr.overall[(gr.asin == item) & (gr.reviewerID == user2)].values[0])
        dict_item_vote[item]=list_votes
    return dict_item_vote


def pearson_def(x, y):
    dict_corr = cor_list(x,y)
    avg_x = avg_vot(x)
    avg_y = avg_vot(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for key,value in dict_corr.items():
        if len(value)>0:
            xlist= value[0]
            ylist = value[1]
            xdiff = xlist - avg_x
            ydiff = ylist - avg_y
            diffprod += xdiff * ydiff
            xdiff2 += xdiff * xdiff
            ydiff2 += ydiff * ydiff
            
    if(math.sqrt(abs(xdiff2 * ydiff2)) != 0):
        return diffprod / math.sqrt(abs(xdiff2 * ydiff2))
    else:
        return 0
    

def squared(list):
    return [i ** 2 for i in list]


def weight_sim(doc1, doc2):
    dict_wei = cor_list(doc1,doc2)
    
    final_v=0;
    
    for key,value in dict_wei.items():
        if len(value)>0:
            
            xdiff = value[0]
            votes_1 = squared(list(df.overall[df.reviewerID == doc1]))
            diffprod1 = xdiff / math.sqrt(abs(sum(votes_1)))

            ydiff = value[1]
            votes_2 = squared(list(df.overall[df.reviewerID == doc2]))
            diffprod2 = ydiff / math.sqrt(abs(sum(votes_2)))
            
            final_v += diffprod1 * diffprod2
            
    return final_v


def pred_vote(user_x,item_x):
    avg_userx = avg_vot(user_x)
    user_list = list(df.reviewerID[df.asin == item_x])
    
    diffprod = 0
    for i_users in user_list:
        if i_users == user_x:
            continue
        weights = weight_sim(user_x,i_users)
        tou = len(gr.overall[(gr.asin == item_x) & (gr.reviewerID == i_users)].values)
        if  tou > 0:
            vote_i = gr.overall[(gr.asin == item_x) & (gr.reviewerID == i_users)].values[0]
            
        avg_useri = avg_vot(i_users)
        diffprod += weights * (vote_i - avg_useri)
        
    return avg_userx + (0.01 * diffprod)

arj = df[['reviewerID','overall','asin']]
final_users = list(arj.reviewerID.unique())

user_li=[];
prod_li=[];
prediction_list = []
actual_list = []

dict_compare ={}

for us in final_users[:500]:
    pr_values = list(arj.asin[df.reviewerID == us])
    for pr in pr_values[:1]:
        prediction = pred_vote(us,pr)
        orginal_value = arj.overall[(arj.asin == pr) & (arj.reviewerID == us)].values[0]
        prediction_list.append(prediction)
        actual_list.append(orginal_value)
        user_li.append(us)
        prod_li.append(pr)
        dict_compare[pr,us]=[prediction,orginal_value]
        
percentile_list = pd.DataFrame({'Product': prod_li,'user': user_li,'Prediction': prediction_list,'Orginal': actual_list})

from sklearn.metrics import mean_absolute_error
mean_absolute_error(prediction_list, actual_list)

pd.DataFrame({'Product': prod_li,'user': user_li,'Prediction': prediction_list,'Orginal': actual_list}).to_csv('C:\\Users\\SarathKumar\\Desktop\\dd.csv')



