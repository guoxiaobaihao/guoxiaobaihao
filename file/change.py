import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import sklearn.multioutput.MultiOutputRegressor as MultiOutputRegressor
def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_squared_error(label,preds)
    return 'myFeval',score

def threeto(lane):
    line_new = lane[:3]
    num_list = [float(ss) for ss in lane[3:]]

    if (num_list[1] + num_list[3] + num_list[5]) > 0:
        sumall = (num_list[0] * num_list[1] + num_list[2] * num_list[3] + num_list[4] * num_list[5]) / (
                    num_list[1] + num_list[3] + num_list[5])
    else:
        sumall = np.mean([num_list[0] + num_list[2] + num_list[4]])
    line_new.append(sumall)
    return line_new

def onetime(vdfive,train_vd_new,train_vd_DB):
    vdfive_new = []
    gry = ['D01', 'D03', 'D06', 'D09', 'D12']
    for da in vdfive:#
        med = [da,'1']
        nozero = []
        for gg in gry:
            sear_1 = [gg, da, '1']
            if sear_1 in train_vd_DB:
                list_id = train_vd_DB.index(sear_1)
                find_num = train_vd_new[list_id][3]
                if find_num>0:
                    nozero.append(find_num)
                #med.append(find_num)
                #else:
                    #continue
            else:
                continue
        try:
            meanget = np.mean(nozero)
        except:
            meanget = 0
        if meanget <1:
            continue
        for gg in gry:
            sear_1 = [gg, da, '1']
            if sear_1 in train_vd_DB:
                list_id = train_vd_DB.index(sear_1)
                find_num = train_vd_new[list_id][3]
                if find_num>0:
                    med.append(find_num)
                else:
                    med.append(meanget)
                #med.append(find_num)
                #else:
                    #continue
            else:
                continue

        if len(med) == 7:
            vdfive_new.append(med)

    return vdfive_new

train_vd = []
train_vd_DB = []

with open('E:\\21down\\DC\\data\\给选手的数据\\train_data_vd.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        if f_csv.line_num == 1:
            print(row)
            head = row
            continue
        if row[2]=='1':
            train_vd.append(row[:-2])
            train_vd_DB.append(row[:3])
print(train_vd[0],train_vd_DB[0])

train_vd_new = []
for st in train_vd:
    train_vd_new.append(threeto(st))

train_time = []
for ss in train_vd_DB:
    if ss[1] not in train_time:
        train_time.append(ss[1])
print(train_time[:10],len(train_time))

train_vd_nowe = onetime(train_time,train_vd_new,train_vd_DB)
print(train_vd_nowe[:2],len(train_vd_nowe),'zheli')
ylabel_time = []
ylabel = []

with open('E:\\21down\\DC\\data\\给选手的数据\\train_data_gantry_sep_1.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        if f_csv.line_num == 1:
            print(row)
            head = row
            continue
        ylabel.append(row)
        ylabel_time.append(row[0])
print(ylabel_time[0],ylabel[0])

xy_com = []
for jj in train_vd_nowe:
    if jj[0] in ylabel_time:
        list_id = ylabel_time.index(jj[0])
        y_num = float(ylabel[list_id][1])
        xy_com.append(jj+[y_num])
print(len(xy_com),xy_com[:2],len(train_vd_nowe))

x_sample = []
y_sample = []
for xy in xy_com:
    x_sample.append(xy[2:7])
    y_sample.append([xy[7],xy[7]+5])
print(x_sample[0],y_sample[0])
x_sample = np.array(x_sample)
y_sample = np.array(y_sample)
train_dataset, test_dataset, train_labels, test_labels = train_test_split(x_sample, y_sample, test_size=0.1, random_state=1)
print(len(train_dataset),len(test_dataset),len( train_labels),len(test_labels))
print(train_dataset[0], test_dataset[0], train_labels[0], test_labels[0])
'''
with open('./form/onewithfive_1.csv','w',newline='')as f:
    f_csv = csv.writer(f)
    head = ['BegginTime','direction','vd_1','vd_2','vd_3','vd_4','vd_5',
            'TransTime'
           ]
    f_csv.writerow(head)
    f_csv.writerows(xy_com)
'''
'''
xgb1 = XGBRegressor()
parameters = {
    'objective': ['reg:linear'],
    'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05],
    'max_depth': [2, 3, 4, 5, 6],
    'min_child_weight': [4],
    'silent': [1],
    'subsample': [0.7],
    'colsample_bytree': [0.7],
    'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv=2,
                        n_jobs=5,
                        verbose=True)

xgb_grid.fit(train_dataset, train_labels)
print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
'''
'''
#xgboost = XGBRegressor( colsample_bytree=0.7, learning_rate = 0.02 ,max_depth=4, min_child_weight=4, n_estimators=500, subsample=0.7,feval = myFeval) #learning_rate = 0.03
xgboost = XGBRegressor( colsample_bytree=0.7, learning_rate = 0.02 ,max_depth=4, min_child_weight=4, n_estimators=500, subsample=0.7) #learning_rate = 0.03
xgboost.fit(train_dataset,train_labels)
y_test_labels = xgboost.predict( test_dataset)
y_train_labels = xgboost.predict(train_dataset)
print(len(y_test_labels))
print(len(y_test_labels[0]))
out = mean_squared_error(y_train_labels,train_labels)
print(out)
out1 = mean_squared_error(y_test_labels,test_labels)
print(out1)
'''
# fitting  MultiOutputRegressor(xgb.XGBRegressor(objective='reg:linear')).fit(X, y)
multioutputregressor = MultiOutputRegressor(XGBRegressor( colsample_bytree=0.7, learning_rate = 0.02 ,max_depth=4, min_child_weight=4, n_estimators=500, subsample=0.7,objective='reg:linear',feval = myFeval)).fit(train_dataset,train_labels)

# predicting
print(np.mean((multioutputregressor.predict(train_dataset) - np.array(train_labels))**2, axis=0) ) # 0.004, 0.003, 0.005)

#把数据整合成一个小时的那种
def onehour(X,Y):
    new_X = []
    new_Y = []
    for sh in range(len(Y)-12):
        new_X.append(X[sh:sh+12,:].reshape(1,-1)[0])
        new_Y.append(Y[sh:sh + 12, :].reshape(1, -1)[0])
    return new_X,new_Y

def cutneg(xy):


    return new_xy


