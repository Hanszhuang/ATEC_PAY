

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pyplot as plt
```


```python
train = pd.read_csv('../data/atec_anti_fraud_train.csv')
test = pd.read_csv('../data/atec_anti_fraud_test_a.csv')
```


```python
train = train[train.label!=-1]

```


```python
dtrain = train[train.date<=20171005]
dvalid = train[train.date>20171005]
```


```python
trainset = lgb.Dataset(dtrain.drop(['id','date','label'],axis=1),label=dtrain.label)
validset = lgb.Dataset(dvalid.drop(['id','date','label'],axis=1),label=dvalid.label)
```


```python
lgb_params =  {
    'boosting_type': 'gbdt',
    'objective': 'binary',
#    'metric': ('multi_logloss', 'multi_error'),
    #'metric_freq': 100,
    'is_training_metric': False,
    'min_data_in_leaf': 12,
    'num_leaves': 32,
    'learning_rate': 0.07,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'verbosity':-1,
#    'gpu_device_id':2,
#    'device':'gpu'
#    'lambda_l1': 0.001,
#    'skip_drop': 0.95,
#    'max_drop' : 10
    #'lambda_l2': 0.005
    #'num_threads': 18
}    
```


```python
def evalMetric(preds,dtrain):
    label = dtrain.get_label()
    fpr,tpr,threholds= metrics.roc_curve(label, preds, pos_label=1)
    res = 0.0
    ths= [0.001,0.005,0.01]
    weight = [0.4,0.3,0.3]
    for wi,th in enumerate(ths):
        index = 1
        for index in range(1,len(fpr)):
            if fpr[index-1] <=th and fpr[index]>=th:
                res = res+(tpr[index-1] + tpr[index])/2 * weight[wi]

    
    return 'res',res,True
```

### 线下部分


```python
model =lgb.train(lgb_params,trainset,feval=evalMetric,early_stopping_rounds=100,verbose_eval=5,valid_sets=[validset],num_boost_round=10000)
```

    Training until validation scores don't improve for 100 rounds.
    [5]	valid_0's binary_logloss: 0.437999	valid_0's res: 0.425764
    [10]	valid_0's binary_logloss: 0.294353	valid_0's res: 0.441351
    [15]	valid_0's binary_logloss: 0.205692	valid_0's res: 0.455489
    [20]	valid_0's binary_logloss: 0.148318	valid_0's res: 0.468707
    [25]	valid_0's binary_logloss: 0.110154	valid_0's res: 0.479526
    [30]	valid_0's binary_logloss: 0.0842264	valid_0's res: 0.485981
    [35]	valid_0's binary_logloss: 0.066514	valid_0's res: 0.491185
    [40]	valid_0's binary_logloss: 0.054427	valid_0's res: 0.494313
    [45]	valid_0's binary_logloss: 0.0460528	valid_0's res: 0.498733
    [50]	valid_0's binary_logloss: 0.0402539	valid_0's res: 0.503485
    [55]	valid_0's binary_logloss: 0.036113	valid_0's res: 0.50651
    [60]	valid_0's binary_logloss: 0.0332348	valid_0's res: 0.506384
    [65]	valid_0's binary_logloss: 0.0311578	valid_0's res: 0.509773
    [70]	valid_0's binary_logloss: 0.0297455	valid_0's res: 0.509821
    [75]	valid_0's binary_logloss: 0.0290196	valid_0's res: 0.508855
    [80]	valid_0's binary_logloss: 0.0286838	valid_0's res: 0.507318
    [85]	valid_0's binary_logloss: 0.0283068	valid_0's res: 0.505877
    [90]	valid_0's binary_logloss: 0.0278047	valid_0's res: 0.507239
    [95]	valid_0's binary_logloss: 0.0275878	valid_0's res: 0.507778
    [100]	valid_0's binary_logloss: 0.0286192	valid_0's res: 0.477237
    [105]	valid_0's binary_logloss: 0.0298978	valid_0's res: 0.448471
    [110]	valid_0's binary_logloss: 0.0295851	valid_0's res: 0.450341
    [115]	valid_0's binary_logloss: 0.042211	valid_0's res: 0.248535
    [120]	valid_0's binary_logloss: 0.0407282	valid_0's res: 0.266846
    [125]	valid_0's binary_logloss: 0.0457676	valid_0's res: 0.225788
    [130]	valid_0's binary_logloss: 0.0452163	valid_0's res: 0.235704
    [135]	valid_0's binary_logloss: 0.0832717	valid_0's res: 0.0364803
    [140]	valid_0's binary_logloss: 0.0714189	valid_0's res: 0.0736892
    [145]	valid_0's binary_logloss: 0.0690056	valid_0's res: 0.0866783
    [150]	valid_0's binary_logloss: 0.0653055	valid_0's res: 0.101077
    [155]	valid_0's binary_logloss: 0.0639453	valid_0's res: 0.108459
    [160]	valid_0's binary_logloss: 0.0631818	valid_0's res: 0.113084
    [165]	valid_0's binary_logloss: 0.0620963	valid_0's res: 0.119088
    Early stopping, best iteration is:
    [66]	valid_0's binary_logloss: 0.0307962	valid_0's res: 0.511128


## 线上预测


```python
all_trainset = lgb.Dataset(train.drop(['id','date','label'],axis=1),label=train.label)
```


```python
model =lgb.train(lgb_params,all_trainset,verbose_eval=5,valid_sets=[all_trainset],num_boost_round=65,feval=evalMetric)
```

    [5]	training's binary_logloss: 0.437337	training's res: 0.498131
    [10]	training's binary_logloss: 0.293253	training's res: 0.513191
    [15]	training's binary_logloss: 0.204311	training's res: 0.52439
    [20]	training's binary_logloss: 0.14675	training's res: 0.536875
    [25]	training's binary_logloss: 0.108384	training's res: 0.544943
    [30]	training's binary_logloss: 0.082211	training's res: 0.554686
    [35]	training's binary_logloss: 0.0643196	training's res: 0.560865
    [40]	training's binary_logloss: 0.0520004	training's res: 0.567934
    [45]	training's binary_logloss: 0.0433078	training's res: 0.573783
    [50]	training's binary_logloss: 0.0372245	training's res: 0.579649
    [55]	training's binary_logloss: 0.032896	training's res: 0.585514
    [60]	training's binary_logloss: 0.0298652	training's res: 0.588566
    [65]	training's binary_logloss: 0.0276897	training's res: 0.591338



```python
pred=model.predict(test.drop(['id','date'],axis=1))
```


```python
res =pd.DataFrame({'id':test.id,'score':pred})
res.to_csv('../result/lgb-0.511128-boost-65.csv',index=False)
```
