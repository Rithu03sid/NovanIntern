import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
data_fake=pd.read_csv('C:\Users\Admin\Desktop\Project 2\Fake.csv')
data_true=pd.read_csv('C:\Users\Admin\Desktop\Project 2\True.csv')
data_fake.head()
data_true.tail()
data_fake["class"]=0
data_true['class']=1
data_fake.shape, data_true.shape
data_fake_manual_testing = data_fake.tail(10)
for i in range(23480,23470,-1):
    data_fake.drop([i],axis = 0, inplace = True)


data_true_manual_testing = data_true.tail(10)
for i in range(21416,21406,-1):
    data_true.drop([i],axis = 0, inplace = True)

data_fake.shape, data_true.shape
data_fake_manual_testing['class']=0
data_true_manual_testing['class']=1
data_fake_manual_testing.head(10)
data_merge=pd.concat([data_fake, data_true], axis = 0)
data_merge.head(10)
data_merge.columns
data=data_merge.drop(['title','subject','date'], axis = 1)
data.head()
data.isnull().sum()
data = data.sample(frac = 1)
data.head()
data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)
data.columns
data.head()
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+',b'',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    return text
data['text'] = data['text'].apply(wordopt)
x = data['text']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.50)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
LR.score(xv_test, y_test)
print (classification_report(y_test, pred_lr))
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)
print (classification_report(y_test, pred_lr))
from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train, y_train)
pred_gb = GB.predict(xv_test)
GB.score(xv_test, y_test)
print(classification_report(y_test, pred_gb))
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train, y_train)
pred_rf = RF.predict(xv_test)
RF.score(xv_test, y_test)
print (classification_report(y_test, pred_rf))
def output_lable(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "True News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    return print("\n\nLR Predicition: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction:{}".format(output_lable(pred_LR[0]),
                                                                                                             output_lable(pred_DT[0]),
                                                                                                             output_lable(pred_GB[0]),

news = str(input()) 
manual_testing(news)
news = str(input()) 
manual_testing(news)