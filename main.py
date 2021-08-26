# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#1.建立一个初步的模型，看一下效果怎么样。
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)#经验值
xgb_param = xgb1.get_xgb_params()#得到模型的参数
xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)#转换成原生xgboost需要的数据格式。
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
        metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)#注意啊这是原生的模型自带的，你只需传参，cv_folds=5，表示5%的数据用于交叉验证。这个cvresults返回训练集和测试集的误差，行数就是最大迭代的次数。
xgb1.set_params(n_estimators=cvresult.shape[0])
xgb1.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
dtrain_predictions = alg.predict(dtrain[predictors])#算准确率用的
dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]#算auc用的
feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)#算重要度的指标
#‘weight’ - the number of times a feature is used to split the data across all trees.‘gain’ - the average gain of the feature when it is used in trees.‘cover’ - the average coverage of the feature when it is used in trees.
#weight - 该特征在所有树中被用作分割样本的特征的次数。gain - 在所有树中的平均增益。cover - 在树中使用该特征时的平均覆盖范围。
#
#2开始按步骤调参用的XGBclassifer。
#第一步调优
params_test1 = {'n_estimators': [400, 500, 600, 700, 800]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 27,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = XGBClassfoer(**other_params)
optimized_XGB 1= GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
optimized_XGB.fit(X_train, y_train)
evalute_result = optimized_GBM.grid_scores_
##第二步调优
param_test2 = { 'max_depth':range(3,10,2), 'min_child_weight':range(1,6,2)}
optimized_XGB2= GridSearchCV(estimator = XGBClassifier(         learning_rate =0.1, n_estimators=140, max_depth=5,
min_child_weight=1, gamma=0, subsample=0.8,             colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4,     scale_pos_weight=1, seed=27),  param_grid = param_test2,     scoring='roc_auc',n_jobs=4,iid=False, cv=5)
optimized_XGB2.fit(train[predictors],train[target])
optimized_XGB2.grid_scores_, optimized_XGB2.best_params_,     optimized_XGB2.best_score_
#依次类推，得到最后的最优参数集合，再建立模型用于预测
#3.最终模型
model = XGBClassifer(learning_rate=0.1, n_estimators=550, max_depth=4, min_child_weight=5, seed=27,
                             subsample=0.7, colsample_bytree=0.7, gamma=0.1, reg_alpha=1, reg_lambda=1)
model.fit(X_train, y_train)
ans = model.predict(X_test)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
