
# coding: utf-8

# Done by: Raphael D. Seng (1001744, raphael_seng@mymail.sutd.edu.sg), Jordan Sim (100510, jordan_sim@mymail.sutd.edu.sg)
# 
# Results are in the zip folder. 

# In[1]:


import pandas as pd
import numpy as np


# ### Part 2

# In[2]:


def getDF(name_of_file):
    l_list = []
    with open(name_of_file) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]
            if len(inner_list)==2:
                l_list.append(inner_list)
    df=pd.DataFrame(l_list,columns=['x','y']) 
    return df


# #### 2) a) b)

# In[3]:


def getEmission(name_of_file,k=1.0):
    df=getDF(name_of_file)
    df_counts= pd.DataFrame(df['y'].value_counts()).reset_index().rename(columns={'index':'y','y':'total'})
    df=df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'count'})
    df_2= pd.merge(df, df_counts, how='left', on='y')

    #adding in #UNK# part   
    df_2['b'] = df_2.apply(lambda row: k/(row['total']+k) 
                           if row['x']=='#UNK#' else float(row['count'])/(row['total']+k), axis=1)
    tag_list = list(df_2['y'].unique())
    for state in tag_list:
        unk_total = df_2.loc[(df_2.y==state),'total'].iloc[0]
        df_2= df_2.append(pd.DataFrame(np.array([['#UNK#', state, k, unk_total, k/(k+unk_total)]]), columns=['x','y','count','total','b']) ,ignore_index=True)

    return df_2


# In[4]:


train_EN=getEmission('EN/train')
train_CN=getEmission('CN/train')
train_ES=getEmission('ES/train')
train_RU=getEmission('RU/train')
train_EN


# In[150]:


display(train_EN.loc[train_EN['x']=='#UNK#',:])
# print list(train_EN['y'].unique())


# The getTop_E( ) function assigns each word with one tag that gives the best emission probability

# In[6]:


def getTop_E(df):
    df["rank"] = df.groupby("x")["b"].rank(method="max", ascending=False)
    df= df.loc[df['rank']==1.0].drop(['rank'], axis=1).reset_index(drop=True)
    return df


# #### 2c)

# In[7]:


def predEmissionOnly(devin,train,k=1.0):
    train=getTop_E(getEmission(train))
    l_list=[]
    with open(devin) as f:
        for line in f:
            l_list.append(line.strip())
    df_in=pd.DataFrame(l_list,columns=['x'])

    #predict y using training set
    df1=pd.merge(df_in,train, how='left', on='x').rename(columns={'y':'pred_y'})

    #get y* when x is #UNK#
    y_star=df1.loc[df1['total']==df1['total'].min(),'pred_y'].iloc[0]
    df1.loc[(df1['x'] != '') & (df1['pred_y'].isnull()), 'pred_y'] = y_star
    df1['pred_y']=df1['pred_y'].fillna('')
    
    # export dev.p2.out
    df1[['x','pred_y']].to_csv(lang+'/dev.p2.out', header=None, index=None, sep=' ')
    
    return lang+' Done'


# In[8]:


lang_list=['EN','CN','ES','RU']
for lang in lang_list:
    print predEmissionOnly(lang+'/dev.in',lang+'/train',lang+'/dev.out')


# ### Part 3

# In[9]:


def readDevIn(devin, train): # read dev.in file to be used for viterbi algorithm
    l_list=[]
    with open(devin) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]
            if len(inner_list)>2:
                inner_list[:-1] = [''.join(inner_list[:-1])]
            l_list.append(inner_list[0])
    Lsub = []
    L2=[]
    for e in l_list:
        if e=='':
            if Lsub: 
                L2.append(Lsub)
            Lsub = [e]
        else:
            Lsub.append(e)
    L2.append(Lsub)
    for tweet in L2:
        if tweet[0]=='':
            tweet[0]='START'
        else:
            tweet.insert(0, 'START')
        if tweet[-1]=='':
            tweet[-1]='STOP'
        else:
            tweet.append('STOP')

    return L2
    
# display(readDevIn('EN'+'/dev.in',train_EN))


# In[10]:


def get_Y_IN_ORDER(name_of_file):
    l_list = ['']
    with open(name_of_file) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]
            if len(inner_list)>2:
                inner_list[:-1] = [''.join(inner_list[:-1])]
            l_list.append(inner_list[len(inner_list)-1])
    return l_list

y_in_order=get_Y_IN_ORDER('EN/train')


# In[11]:


tag_list=list(train_EN['y'].unique())

def transitionCounter(name_of_file):
    y_in_order=get_Y_IN_ORDER(name_of_file)
    transition={'START': {'START':0.0,'O':0.0,'B-negative':0.0,'B-neutral':0.0,'B-positive':0.0,'I-negative':0.0, 'I-neutral':0.0,'I-positive':0.0,'STOP':0.0}, 'O': {'START':0.0,'O':0.0,'B-negative':0.0,'B-neutral':0.0,'B-positive':0.0,'I-negative':0.0, 'I-neutral':0.0,'I-positive':0.0,'STOP':0.0}, 'B-negative': {'START':0.0,'O':0.0,'B-negative':0.0,'B-neutral':0.0,'B-positive':0.0,'I-negative':0.0, 'I-neutral':0.0,'I-positive':0.0,'STOP':0.0}, 'B-neutral': {'START':0.0,'O':0.0,'B-negative':0.0,'B-neutral':0.0,'B-positive':0.0,'I-negative':0.0, 'I-neutral':0.0,'I-positive':0.0,'STOP':0.0}, 'B-positive': {'START':0.0,'O':0.0,'B-negative':0.0,'B-neutral':0.0,'B-positive':0.0,'I-negative':0.0, 'I-neutral':0.0,'I-positive':0.0,'STOP':0.0}, 'I-negative': {'START':0.0,'O':0.0,'B-negative':0.0,'B-neutral':0.0,'B-positive':0.0,'I-negative':0.0, 'I-neutral':0.0,'I-positive':0.0,'STOP':0.0}, 'I-neutral': {'START':0.0,'O':0.0,'B-negative':0.0,'B-neutral':0.0,'B-positive':0.0,'I-negative':0.0, 'I-neutral':0.0,'I-positive':0.0,'STOP':0.0}, 'I-positive': {'START':0.0,'O':0.0,'B-negative':0.0,'B-neutral':0.0,'B-positive':0.0,'I-negative':0.0, 'I-neutral':0.0,'I-positive':0.0,'STOP':0.0}, 'STOP': {'START':0.0,'O':0.0,'B-negative':0.0,'B-neutral':0.0,'B-positive':0.0,'I-negative':0.0, 'I-neutral':0.0,'I-positive':0.0,'STOP':0.0}}
    for i in range(0,len(y_in_order)-1):
        if y_in_order[i] == '': #START is indicated by ''
            transition['START'][y_in_order[i+1]]+=1.0/y_in_order[:len(y_in_order)-1].count("")
        elif y_in_order[i+1] == '': #STOP is indicated by ''
            transition[y_in_order[i]]['STOP']+=1.0/y_in_order.count(y_in_order[i])
        else: # y(i-1) -> y transition
            transition[y_in_order[i]][ y_in_order[i+1]] +=1.0/y_in_order.count(y_in_order[i])
    return transition


# In[12]:


get_ipython().run_cell_magic(u'time', u'', u"a_dict=transitionCounter('EN/train')")


# In[13]:


def getY_List(N,y_list,pi,df_a,states):
    for i in range(N-2,0,-1):       
        d={}
        for u in range(1,8):
            if i==N-2:
                x= pi[N-2][u]+loG(a_dict[states[u]]['STOP'])
            else:
                x= pi[i][u]+loG(a_dict[states[u]][y_list[i+1]])
            d[states[u]]=x
        y_list[i]=max(d, key=d.get)
    return y_list


# Define own log function to solve numerical underflow issue. np.log(0) gives very long runtime since undefined, so manually assign a large negative number

# In[14]:


def loG(x):
    if x==0.0:
        return -(999e10)
    else:
        return np.log(x)


# The get_b() function returns the emission probability given the word and the tag/state.

# In[15]:


def get_b(d,word,tag): 
    if word in d.keys():
        d1=d.get(word)
        if tag in d1.keys():
            b=float(d1.get(tag))
        else:
            b=0.0
    else:
        b=float(d['#UNK#'][tag])
    return b


# In[16]:


def emission_df_to_dict(df):
    return df.groupby('x').apply(lambda x: dict(zip(x.y, x.b))).to_dict()


# The viterbi() function does the forward recursion and outputs the dev.out file. It calls the getY_list() function to do backtracking and generate the optimal sequence of sentiments.
# 
# At the 'STOP' layer, pi(k,'STOP') is not calculated since it is not needed. 

# In[17]:


train_dict={'EN':emission_df_to_dict(train_EN),'ES':emission_df_to_dict(train_ES),'CN':emission_df_to_dict(train_CN),'RU':emission_df_to_dict(train_RU)}
tag_list=list(train_EN['y'].unique())

def viterbi(devin,train,a_dict):
    #reading data
    b_dict=train_dict[train[:2]]
    states=['START']+tag_list+['STOP']
    #read devin as one list with innerlists representing each tweet
    devin_list=readDevIn(devin,train)
    print "No. of tweets: "+str(len(devin_list))
    Y=[]
    for tweet in devin_list:
        N=len(tweet)
        y_list=['']*N
        y_list[0]='START'
        y_list[-1]='STOP'
        keys = range(N)
        pi={key: [np.nan]*9 for key in keys}
        for v in range(9):
            if v==0:
                pi[0][v]=loG(1.0)
            else:
                pi[0][v]=loG(0.0)
        #forward recursion
        for k in range(1,N):
            word=tweet[k]
            if k==1:
                for u in range(1,8):
                    b=get_b(b_dict,word,states[u])
                    pi[k][u]=loG(a_dict['START'][states[u]])+loG(b)
            elif 1< k <N-1:
                for v in range(1,8):
                    list1=[]
                    b=get_b(b_dict,word,states[v])
                    for u in range(1,8):         
                        x=pi[k-1][u]+loG(a_dict[states[u]][states[v]])+loG(b) 
                        list1.append(x)
                    pi[k][v]= max(list1)
        #decoding
        y_list=getY_List(N,y_list,pi,a_dict,states)
        Y+=y_list 
    print "Generating output file..."
    
    #add words and predicted tags to sentence in dev.p3.out
    Y = [e for e in Y if e != 'START']
    YY = [x if x !='STOP' else '' for x in Y]

    L3=[]
    for tweet in devin_list: 
        tweet.remove('START')
        tweet[-1]=''
        L3+=tweet
    df1=pd.DataFrame(L3,columns=['x'])
    df2=pd.DataFrame(YY,columns=['y'])
    df_out=pd.concat([df1,df2],axis=1)   
    df_out[['x','y']].to_csv(lang+'/dev.p3.out', header=None, index=None, sep=' ')

    display(df_out)
    
    return lang+' Done'


# The following part is to return dev.out files for the different languages

# In[18]:


get_ipython().run_cell_magic(u'time', u'', u"lang_list=['EN','CN','ES','RU']\nfor lang in lang_list:\n    a_dict=transitionCounter(lang+'/train')\n    print viterbi(lang+'/dev.in',lang+'/train',a_dict)")


# ### Part 4
# 
# We used Beam Search as part of intuition for this part. 
# 
# The idea is to store each cell in the pi matrix with a list of the top 5 possible values with their respective tag from the previous layer. 
# For example pi(k,v) is a list containing top 5 of tag , value pair of pi(k-1,tag)*a(tag,v)*bv(word). Where tag is from previous layer. 
# 
# As the forward recursion is done, at each node, the top 5 best pi value is always stored, from 1st layer til last layer. 
# 
# In the backtracking, start from last layer and get top 5 scores and respective tag. Each tag is part of 5 possible paths. From this paths, transition back to 2nd last layer. There are many possible paths now with two tags and each has a score. Again, select top 5 paths based on score. 
# 
# Continue backtracking til first layer and get top 5 paths. The 5th best path will the one has the 5th best overall score out of these paths. 
# 
# topk is number of values & number of paths to stores. 
# kbest is which ranking to return.

# In[41]:


def kbest_viterbi(devin,train,a_dict,kbest,topk=5):    
    #reading data
    b_dict=train_dict[train[:2]]
    states=['START']+tag_list+['STOP']
    #read devin as one list with innerlists representing each tweet
    devin_list=readDevIn(devin,train)
    print "No. of tweets: "+str(len(devin_list))
    Y=[]
    for tweet in devin_list:
        N=len(tweet)
        y_list=['']*N
        y_list[0]='START'
        y_list[-1]='STOP'
        keys = range(N)
        pi={key: [np.nan]*9 for key in keys}
        for v in range(9):
            if v==0:
                pi[0][v]=loG(1.0)
            else:
                pi[0][v]=loG(0.0)
        #forward recursion 
        for k in range(1,N):
            word=tweet[k]
            if k==1:
                for u in range(1,8):
                    b=get_b(b_dict,word,states[u])
                    pi[k][u]=[loG(a_dict['START'][states[u]])+loG(b)]
            elif 1< k <N-1:
                for v in range(1,8):
                    list1=[]
                    for u in range(1,8):
                        node_list=pi[k-1][u]
                        b=get_b(b_dict,word,states[v])
                        for val in node_list:
                            x=val+loG(a_dict[states[u]][states[v]])+loG(b)
                            list1.append(x)
                    pi[k][v]= sorted(list1,reverse=True)[:topk]   #get top 5 values
        #decoding
        y_list=get_kbest_Y_List(N,y_list,pi,a_dict,states,kbest,topk)
        Y+=y_list 
    print "Generating output file..."
    
    #add words and predicted tags to sentence in dev.p3.out
    Y = [e for e in Y if e != 'START']
    YY = [x if x !='STOP' else '' for x in Y]

    L3=[]
    for tweet in devin_list: 
        tweet.remove('START')
        tweet[-1]=''
        L3+=tweet
    df1=pd.DataFrame(L3,columns=['x'])
    df2=pd.DataFrame(YY,columns=['y'])
    df_out=pd.concat([df1,df2],axis=1)   
    df_out[['x','y']].to_csv(lang+'/dev.p4.out', header=None, index=None, sep=' ')

    display(df_out)
    
    return lang+' Done'


# In[42]:


def get_kbest_Y_List(N,y_list,pi,a_dict,states,kbest,topk):
    path={}
    for k in range(topk):
        path[k]=[]
        
    #decoding
    for i in range(N-2,0,-1): 
        d=[]
        for u in range(1,8):
            node_list=pi[i][u]
            for val in node_list:
                if i==N-2:
                    v='STOP'
                    score= val+loG(a_dict[states[u]][v])
                    d.append([v,states[u],score])
                else:
                    for v in newstates:
                        score= val+loG(a_dict[states[u]][v])
                        d.append([v,states[u],score])
        L=sorted(d,key=lambda sublist: sublist[-1], reverse=True)[:topk] #list of states & scores
        newstates=[]
        for i in range(len(L)):
            path[i].append(L[i])
            if L[i][1] not in newstates:
                newstates.append(L[i][1])
                
    #get path and convert into list/sequence of tags            
    try:
        boo_list=[]
        for i in range(topk):
            #get last element in the top 5 paths, boo is the score of the overall path
            boo=path.get(i)[-1][2]
            boo_list.append(boo)
#             printing next line will give score and top 5 optimal paths 
#             print boo, [x[:2][1] for x in path.get(i)][::-1]
        B_list=sorted(boo_list,reverse=True)
        B=B_list[kbest-1]
        
        #get k-th best path
        for i in range(topk):
            if B==0.0:
                yylist=[x[:2][1] for x in path.get(kbest-1)][::-1]
            elif B==path.get(i)[-1][2]:
                yylist=[x[:2][1] for x in path.get(i)][::-1]
    except:
        yylist=y_list
    y_list=['START']+yylist+['STOP']
    return y_list


# In[43]:


get_ipython().run_cell_magic(u'time', u'', u"lang_list=['ES','RU']\nfor lang in lang_list:\n    a_dict=transitionCounter(lang+'/train')\n    print kbest_viterbi(lang+'/dev.in',lang+'/train',a_dict,kbest=5)")


# How we validated our k best viterbi was to generate 1st best (kbest=1) out of top 5 paths using the same functions. The results generated is almost same as part dev.p3.out file.

# ### Part 5
# 
# We used our Viterbi algorithm from part 3 as our basis to make modifications on.  
# 
# Mod 1) For a start, we made all words to lowercase. This eliminates any differences between same words with different letter cases. For example: “Netflix” and “netflix” will now be the considered the same word in our algorithm.
# 
# Entity = O, I, B. Sentiment =  negative, neutral, positive.
# 
# Mod 2) The main modification was done by splitting up the prediction of Entity and Sentiment. We trained 2 different models using our part 3 Viterbi algorithm. One model is based on just the Entity alone and the other based on just the Sentiment alone. We then do a separate prediction of the entity and sentiment based on the models, subsequently combining them as the final step. 
# 
# The logic for combining the entity and sentiment prediction is shown as such:
# ```
# for any word:
#     if predicted_entity is I or B:
#         if predicted_sentiment is NONE:
#             final_predicted_sentiment = neutral ###sets sentiment as neutral if no sentiment available
#         else:
#           final_predicted_sentiment = predicted_sentiment
#     else: ###predicted entity is O:
#         final_predicted_sentiment  = NONE
# ```

# Mod 3) Change formula of emission for unk to make it homogenous across all tags.

# $$e\left(x\mid y_i\right) = \frac{k}{k+\sum_{i=1}^{n}\text{Count}\left(y_i\right)}, \quad \text{where }n\text{ is number of tags}$$

# In[129]:


def getDF_entity(name_of_file):
    l_list = []
    with open(name_of_file) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]
            if len(inner_list)==2:
                inner_list[0]=inner_list[0].decode('utf-8').lower()
                inner_list[1] = [elt.strip() for elt in inner_list[1].split('-')][0]
                l_list.append(inner_list)
    df=pd.DataFrame(l_list,columns=['x','y']) 
    return df


# In[130]:


def getEmission_entity(name_of_file,k=1.0):
    df=getDF_entity(name_of_file)
    df_counts= pd.DataFrame(df['y'].value_counts()).reset_index().rename(columns={'index':'y','y':'total'})
    df=df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'count'})
    df_2= pd.merge(df, df_counts, how='left', on='y')   
    df_2['b'] = df_2.apply(lambda row: k/(row['total']+k) 
                           if row['x']=='#UNK#' else float(row['count'])/(row['total']+k), axis=1)
    
    #adding in #UNK# part
    count_list=df_2['total'].unique().astype(float)
    count_total=sum(count_list)
    tag_list = list(df_2['y'].unique())
    for state in tag_list:
#         unk_total = df_2.loc[(df_2.y==state),'total'].iloc[0]
        df_2= df_2.append(pd.DataFrame(np.array([['#UNK#', state, k, count_total, 1.0/(k+count_total)]]), columns=['x','y','count','total','b']) ,ignore_index=True)

    return df_2


# In[131]:


def getDF_sentiment(name_of_file):
    l_list = []
    with open(name_of_file) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]
            if len(inner_list)==2:
                inner_list[0]=inner_list[0].decode('utf-8').lower()
                inner_list[1] = [elt.strip() for elt in inner_list[1].split('-')][-1]
                l_list.append(inner_list)
    df=pd.DataFrame(l_list,columns=['x','y']) 
    return df


# In[132]:


train_ES_entity=getEmission_entity('ES/train')
train_RU_entity=getEmission_entity('RU/train')


# In[133]:


def getEmission_sentiment(name_of_file,k=1.0):
    df=getDF_sentiment(name_of_file)
    df_counts= pd.DataFrame(df['y'].value_counts()).reset_index().rename(columns={'index':'y','y':'total'})
    df=df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'count'})
    df_2= pd.merge(df, df_counts, how='left', on='y')
    df_2['b'] = df_2.apply(lambda row: k/(row['total']+k) 
                           if row['x']=='#UNK#' else float(row['count'])/(row['total']+k), axis=1)
    
    #adding in #UNK# part
    count_list=df_2['total'].unique().astype(float)
    count_total=sum(count_list)
    tag_list = list(df_2['y'].unique())
    for state in tag_list:
#         unk_total = df_2.loc[(df_2.y==state),'total'].iloc[0]
        df_2= df_2.append(pd.DataFrame(np.array([['#UNK#', state, k, count_total, 1.0/(k+count_total)]]), columns=['x','y','count','total','b']) ,ignore_index=True)

    return df_2


# In[134]:


train_ES_sentiment=getEmission_sentiment('ES/train')
train_RU_sentiment=getEmission_sentiment('RU/train')


# In[135]:


train_ES_sentiment


# In[136]:


def get_Y_IN_ORDER_entity(name_of_file):
    l_list = ['']
    with open(name_of_file) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]
            if len(inner_list)>2:
                inner_list[:-1] = [''.join(inner_list[:-1])]
            l_list.append(inner_list[len(inner_list)-1].split("-")[0])
    return l_list


# In[137]:


def transitionCounter_entity(name_of_file):
    y_in_order_entity=get_Y_IN_ORDER_entity(name_of_file)
    transition={'START':{'START':0.0, 'O':0.0, 'B':0.0, 'I':0.0,'STOP':0.0},'O':{'START':0.0, 'O':0.0, 'B':0.0, 'I':0.0,'STOP':0.0},'B':{'START':0.0, 'O':0.0, 'B':0.0, 'I':0.0,'STOP':0.0},'I':{'START':0.0, 'O':0.0, 'B':0.0, 'I':0.0,'STOP':0.0},'STOP':{'START':0.0, 'O':0.0, 'B':0.0, 'I':0.0,'STOP':0.0}}

    for i in range(0,len(y_in_order_entity)-1):
        if y_in_order_entity[i] == '': #START is indicated by ''
            transition['START'][y_in_order_entity[i+1]]+=1.0/y_in_order_entity[:len(y_in_order_entity)-1].count("")
        elif y_in_order_entity[i+1] == '': #STOP is indicated by ''
            transition[y_in_order_entity[i]]['STOP']+=1.0/y_in_order_entity.count(y_in_order_entity[i])
        else: # y(i-1) -> y transition
            transition[y_in_order_entity[i]][ y_in_order_entity[i+1]] +=1.0/y_in_order_entity.count(y_in_order_entity[i])
    return transition


# In[138]:


a_dict_entity_ES=transitionCounter_entity('ES/train')
a_dict_entity_RU=transitionCounter_entity('RU/train')


# In[139]:


def get_Y_IN_ORDER_sentiment(name_of_file):
    l_list = ['']
    with open(name_of_file) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]
            if len(inner_list)>2:
                inner_list[:-1] = [''.join(inner_list[:-1])]
            l_list.append(inner_list[len(inner_list)-1].split("-")[-1])
    return l_list


# In[140]:


def transitionCounter_sentiment(name_of_file):
    y_in_order_sentiment=get_Y_IN_ORDER_sentiment(name_of_file)
    transition={'START':{'START':0.0, 'neutral':0.0, 'positive':0.0, 'negative':0.0, 'O':0.0,'STOP':0.0},'neutral':{'START':0.0, 'neutral':0.0, 'positive':0.0, 'negative':0.0, 'O':0.0,'STOP':0.0},'positive':{'START':0.0, 'neutral':0.0, 'positive':0.0, 'negative':0.0, 'O':0.0,'STOP':0.0},'negative':{'START':0.0, 'neutral':0.0, 'positive':0.0, 'negative':0.0, 'O':0.0,'STOP':0.0},'O':{'START':0.0, 'neutral':0.0, 'positive':0.0, 'negative':0.0, 'O':0.0,'STOP':0.0},'STOP':{'START':0.0, 'neutral':0.0, 'positive':0.0, 'negative':0.0, 'O':0.0,'STOP':0.0}}

    for i in range(0,len(y_in_order_sentiment)-1):
        if y_in_order_sentiment[i] == '': #START is indicated by ''
            transition['START'][y_in_order_sentiment[i+1]]+=1.0/y_in_order_sentiment[:len(y_in_order_sentiment)-1].count("")
        elif y_in_order_sentiment[i+1] == '': #STOP is indicated by ''
            transition[y_in_order_sentiment[i]]['STOP']+=1.0/y_in_order_sentiment.count(y_in_order_sentiment[i])
        else: # y(i-1) -> y transition
            transition[y_in_order_sentiment[i]][ y_in_order_sentiment[i+1]] +=1.0/y_in_order_sentiment.count(y_in_order_sentiment[i])
    return transition


# In[141]:


a_dict_sentiment_ES=transitionCounter_sentiment('ES/train')
a_dict_sentiment_RU=transitionCounter_sentiment('RU/train')


# In[142]:


def getY_List_entity(N,y_list,pi,a_dict,states):
    for i in range(N-2,0,-1):       
        d={}
        for u in range(1,4):
            if i==N-2:
                x= pi[N-2][u]+loG(a_dict[states[u]]['STOP'])
            else:
                x= pi[i][u]+loG(a_dict[states[u]][y_list[i+1]])
            d[states[u]]=x
        y_list[i]=max(d, key=d.get)
    return y_list


# In[143]:


def getY_List_sentiment(N,y_list,pi,a_dict,states):
    for i in range(N-2,0,-1):       
        d={}
        for u in range(1,5):
            if i==N-2:
                x= pi[N-2][u]+loG(a_dict[states[u]]['STOP'])
            else:
                x= pi[i][u]+loG(a_dict[states[u]][y_list[i+1]])
            d[states[u]]=x
        y_list[i]=max(d, key=d.get)
    return y_list


# In[144]:


train_dict_entity={'ES':emission_df_to_dict(train_ES_entity),'RU':emission_df_to_dict(train_RU_entity)}
tag_list_entity=['O', 'I', 'B']

train_dict_sentiment={'ES':emission_df_to_dict(train_ES_sentiment),'RU':emission_df_to_dict(train_RU_sentiment)}
tag_list_sentiment=['O', 'neutral', 'positive', 'negative']


# In[147]:


def viterbi_entity_sentiment(devin,train,a_dict_entity,a_dict_sentiment):
    
    #read devin as one list with innerlists representing each tweet
    devin_list=readDevIn(devin,train)
    ##ENTITY
    #reading data
    b_dict_entity=train_dict_entity[train[:2]]
    states_entity=['START']+tag_list_entity+['STOP']

    print "No. of tweets: "+str(len(devin_list))
    print "Entity Prediction"

    Y_entity=[]
    for tweet in devin_list:
        N=len(tweet)
        y_list=['']*N
        y_list[0]='START'
        y_list[-1]='STOP'
        keys = range(N)
        pi_entity={key: [np.nan]*5 for key in keys}
        for v in range(5):
            if v==0:
                pi_entity[0][v]=loG(1.0)
            else:
                pi_entity[0][v]=loG(0.0)
        #forward recursion
        for k in range(1,N):
            word=tweet[k].decode('utf-8').lower()
            if k==1:
                for u in range(1,4):
                    b=get_b(b_dict_entity,word,states_entity[u])
                    pi_entity[k][u]=loG(a_dict_entity['START'][states_entity[u]])+loG(b)
            elif 1< k <N-1:
                for v in range(1,4):
                    list1=[]
                    b=get_b(b_dict_entity,word,states_entity[v])
                    for u in range(1,4):         
                        x=pi_entity[k-1][u]+loG(a_dict_entity[states_entity[u]][states_entity[v]])+loG(b) 
                        list1.append(x)
                    pi_entity[k][v]= max(list1)
        #decoding
        y_list_entity=getY_List_entity(N,y_list,pi_entity,a_dict_entity,states_entity)
        Y_entity+=y_list_entity 
    
    #add words and predicted tags to sentence in dev.p3.out
    Y_entity = [e for e in Y_entity if e != 'START']
    YY_entity = [x if x !='STOP' else '' for x in Y_entity]
    
    
    
    
    print "Sentiment Prediction"

    ##SENTIMENT
    #reading data
    b_dict_sentiment=train_dict_sentiment[train[:2]]
    states_sentiment=['START']+tag_list_sentiment+['STOP']

    Y_sentiment=[]
    for tweet in devin_list:
        N=len(tweet)
        y_list=['']*N
        y_list[0]='START'
        y_list[-1]='STOP'
        keys = range(N)
        pi_sentiment={key: [np.nan]*6 for key in keys}
        for v in range(6):
            if v==0:
                pi_sentiment[0][v]=loG(1.0)
            else:
                pi_sentiment[0][v]=loG(0.0)
        #forward recursion
        for k in range(1,N):
            word=tweet[k].decode('utf-8').lower()
            if k==1:
                for u in range(1,5):
                    b=get_b(b_dict_sentiment,word,states_sentiment[u])
                    pi_sentiment[k][u]=loG(a_dict_sentiment['START'][states_sentiment[u]])+loG(b)
            elif 1< k <N-1:
                for v in range(1,5):
                    list1=[]
                    b=get_b(b_dict_sentiment,word,states_sentiment[v])
                    for u in range(1,5):         
                        x=pi_sentiment[k-1][u]+loG(a_dict_sentiment[states_sentiment[u]][states_sentiment[v]])+loG(b) 
                        list1.append(x)
                    pi_sentiment[k][v]= max(list1)
        #decoding
        y_list_sentiment=getY_List_sentiment(N,y_list,pi_sentiment,a_dict_sentiment,states_sentiment)
        Y_sentiment+=y_list_sentiment
    print "Generating output file..."
    
    #add words and predicted tags to sentence in dev.p3.out
    Y_sentiment = [e for e in Y_sentiment if e != 'START']
    YY_sentiment = [x if x !='STOP' else '' for x in Y_sentiment]
    
    YY=[]
    for counter, entity in enumerate(YY_entity):
        entity_sentiment = entity
        if entity == "B" or entity =="I":
            if YY_sentiment[counter] == 'neutral'or YY_sentiment[counter] =='positive' or YY_sentiment[counter] =='negative':
                entity_sentiment += "-" + YY_sentiment[counter]
            elif YY_sentiment[counter] == 'O':
                entity_sentiment += "-neutral"
        YY.append(entity_sentiment)
    
    L3=[]
    for tweet in devin_list: 
        tweet.remove('START')
        tweet[-1]=''
        L3+=tweet
    df1=pd.DataFrame(L3,columns=['x'])
    df2=pd.DataFrame(YY,columns=['y'])
    df_out=pd.concat([df1,df2],axis=1)  
    if devin[3:6]=='dev':
        df_out[['x','y']].to_csv(lang+'/dev.p5.out', header=None, index=None, sep=' ')
    elif devin[3:7]=='test':
        df_out[['x','y']].to_csv(lang+'/test.p5.out', header=None, index=None, sep=' ')

    display(df_out)
    
    return lang+' Done'


# In[148]:


get_ipython().run_cell_magic(u'time', u'', u"lang='ES'\nprint viterbi_entity_sentiment(lang+'/dev.in',lang+'/train',a_dict_entity_ES,a_dict_sentiment_ES)\n\nlang='RU'\nprint viterbi_entity_sentiment(lang+'/dev.in',lang+'/train',a_dict_entity_RU,a_dict_sentiment_RU)")


# ### Part 5 Testing

# In[149]:


get_ipython().run_cell_magic(u'time', u'', u"lang='ES'\nprint viterbi_entity_sentiment(lang+'/test.in',lang+'/train',a_dict_entity_ES,a_dict_sentiment_ES)\n\nlang='RU'\nprint viterbi_entity_sentiment(lang+'/test.in',lang+'/train',a_dict_entity_RU,a_dict_sentiment_RU)")

