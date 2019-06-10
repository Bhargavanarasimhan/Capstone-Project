#%%%Importing the Necessary libraries
import numpy as np
import pandas as pd
import re
from textblob import TextBlob
#%%%%%%Reading in the Necessary csv files 
ad=["target-address"]
address=pd.read_csv("res_supervised.csv",names=ad)
#%%%%%%%%Initiate the variables
articles = []
fields = ['HD', 'CR', 'WC', 'PD', 'ET', 'SN', 'SC', 'ED', 'PG', 'LA', 'CY', 'LP',
              'TD', 'CT', 'RF', 'CO', 'IN', 'NS', 'RE', 'IPC', 'IPD', 'PUB', 'AN']
factiva=pd.DataFrame(columns=fields) 
#%%%%%Search for the fields in the articles and dump them into a dataframe 
for d in range(len(address)/10):
    with open(address["target-address"].iloc[d], 'r') as infile:
        data = infile.read()
    start = re.search(r'HD\n', data).start()
    for m in re.finditer(r'Document [a-zA-Z0-9]{25}\n', data):
        end = m.end()
        a = data[start:end].strip()
        a = '\n   ' + a
        articles.append(a)
        start = end
    mydata=[]
    for a in articles:
        used = [f for f in fields if re.search(r'\n   ' + f + r'\n', a)]
        unused = [[i, f] for i, f in enumerate(fields) if not re.search(r'\n   ' + f + r'\n', a)]
        fields_pos = []
        for f in used:
            f_m = re.search(r'\n   ' + f + r'\n', a)
            f_pos = [f, f_m.start(), f_m.end()]
            fields_pos.append(f_pos)
        obs = []
        n = len(used)
        for i in range(0, n):
            used_f = fields_pos[i][0]
            start = fields_pos[i][2]
            if i < n - 1:
                end = fields_pos[i + 1][1]
            else:
                end = len(a)
            content = a[start:end].strip()
            obs.append(content)
        for f in unused:
            obs.insert(f[0], '')
        mydata.append(pd.DataFrame(np.array(obs).reshape(1,23),columns=fields))
for i in range(len(mydata)):
    factiva=factiva.append(mydata[i])
#%%%%%%    
factiva.to_csv("Factiva.csv")
#%%%%%Using TextBlob to find the sentiments of the factiva articles
factiva["sentiments"]=""
for i in range(len(factiva)):
    blob=TextBlob(factiva["LP"].iloc[i])
    factiva["sentiments"].iloc[i]=blob.sentiment.polarity
#%%%%%%%%%%Remove the missing values in the sentiments
factiva['HD'].replace('', np.nan, inplace=True)
factiva.dropna(inplace=True)
#%%%Splitting the string to generate the year and month
dates=factiva["PD"]
dates=dates.str.split(" ", expand=True)
#%%%create a columns called factiva month and years
factiva["publication month"]=(dates.iloc[:,1])
factiva["publication year"]=(dates.iloc[:,2])
#%%% find all the months
months=set(factiva["publication month"])
#%%%%%create a dictionary called polarity
mylist=[]
polarity={}
for i in range(1989,2019):
    for j in months:
        newdataframe=pd.DataFrame()
        newdataframe=factiva[factiva['publication year']==str(i)]
        polarity[i,j]=newdataframe.loc[(newdataframe['publication month']==j),'sentiments'].mean()
        
#%%%%%create a polarity Dataframe 
keys=list(polarity.keys())
values=list(polarity.values())
keys=pd.Series(keys)
values=pd.Series(values)
polarity_dataframe=pd.DataFrame()
polarity_dataframe["keys"]=keys
polarity_dataframe["values"]=values
#%%%%fill empty values with zero
polarity_dataframe=polarity_dataframe.fillna(0)
type(polarity_dataframe["keys"].iloc[1])
#%%%%%
year=[]
month=[]
for i in range(len(polarity_dataframe)):
    year.append(polarity_dataframe["keys"].iloc[i][0])
    month.append(polarity_dataframe["keys"].iloc[i][1])
year=pd.Series(year)
month=pd.Series(month)
polarity_dataframe["year"]=year
polarity_dataframe["month"]=month
#%%%%%%%Dumping Polarity dataframe into the csv file
polarity_dataframe.to_csv("polaritydataframe.csv",index=False,header=True)
