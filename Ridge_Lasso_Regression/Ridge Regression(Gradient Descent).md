

```python
# Importing libraries
import numpy as np
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

%matplotlib inline
```


```python
# Load & Quick look the data
house_price = pd.read_csv('kc_house_data.csv')
plt.scatter(house_price['sqft_living'], house_price['price'])
plt.xlabel('sqft_living', fontsize=12)
plt.ylabel('price', fontsize=12)
house_price.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>20141013T000000</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1180</td>
      <td>0</td>
      <td>1955</td>
      <td>0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>20141209T000000</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>2170</td>
      <td>400</td>
      <td>1951</td>
      <td>1991</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>20150225T000000</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>770</td>
      <td>0</td>
      <td>1933</td>
      <td>0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>20141209T000000</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1050</td>
      <td>910</td>
      <td>1965</td>
      <td>0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>20150218T000000</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1680</td>
      <td>0</td>
      <td>1987</td>
      <td>0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




![png](Ridge%20Regression%28Gradient%20Descent%29_files/Ridge%20Regression%28Gradient%20Descent%29_1_1.png)



```python
# converting 'built' and 'renovated' data
house_price['age'] = 2018 - house_price['yr_built']
for index, row in house_price.iterrows():
    if house_price['yr_renovated'][index] == 0:
        house_price.loc[index, 'age_renovated'] = house_price.loc[index, 'age']
    else:
        house_price.loc[index, 'age_renovated'] = 2018 - house_price.loc[index, 'yr_renovated']
```


```python
# Dropping the features we don't need
drop_fields = ['id', 'date', 'long', 'sqft_living15', 'sqft_lot15', 'zipcode', 'lat', 'long', 'grade', 'view', 'yr_built', 'yr_renovated']
house_price = house_price.drop(drop_fields, axis=1)
```


```python
# Check features & data shape after processing.
print ('Shape of data: ' , house_price.shape)
print ('List of features: ', *house_price.columns.values, sep='\n')
```

    Shape of data:  (21613, 12)
    List of features: 
    price
    bedrooms
    bathrooms
    sqft_living
    sqft_lot
    floors
    waterfront
    condition
    sqft_above
    sqft_basement
    age
    age_renovated



```python
# correlation with price feature
print((house_price.corr()['price']).sort_values(ascending=False))
```

    price            1.000000
    sqft_living      0.702035
    sqft_above       0.605567
    bathrooms        0.525138
    sqft_basement    0.323816
    bedrooms         0.308350
    waterfront       0.266369
    floors           0.256794
    sqft_lot         0.089661
    condition        0.036362
    age             -0.054012
    age_renovated   -0.105755
    Name: price, dtype: float64



```python
# split the data set
[train, test] = train_test_split(house_price, test_size= 0.2)
[train, valid] = train_test_split(train, test_size= 0.2)
print('Train :', train.shape, '\nValid: ', valid.shape, '\nTest :', test.shape)
```

    Train : (13832, 12) 
    Valid:  (3458, 12) 
    Test : (4323, 12)



```python
# Splitting data
train_X = train.loc[:, train.columns != 'price']
train_y = train['price']
valid_X = valid.loc[:, valid.columns != 'price']
valid_y = valid['price']
test_X = test.loc[:, test.columns != 'price']
test_y = test['price']
print('X: ', train_X.shape, '\ny: ', train_y.shape)
```

    X:  (13832, 11) 
    y:  (13832,)


## Building Ridge Regression Model


```python
class RidgeRegression(object):
    def __init__(self, learning_rate=1e-5, l2_penalty=1e-1, verbose=False, iteration=1e3):
        self.weights = None
        self.l2_penalty = l2_penalty
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.cost_history = []
        
    def predict(self, X):
        y_pred= np.dot(X, self.weights)
        return(y_pred)

    def calculate_cost(self, y, y_pred):
        cost = np.sum((y - y_pred)**2) + self.l2_penalty*np.sum(self.weights ** 2)
        return cost

    def fit(self, X, y, learning_rate, l2_penalty, iteration, verbose):
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.iteration = iteration
        # Case : 1 feature input data
        if len(X.shape) == 1: 
            self.weights = 0
        else: 
            self.weights = np.zeros(X.shape[1])
        for iter in range(int(iteration)):
            y_pred = self.predict(X)
            error = y - y_pred
            # store cost history for printing
            cost = self.calculate_cost(y, y_pred)
            self.cost_history.append(cost)
            # weight update
            self.weights += learning_rate*((np.dot(X.T, error) - l2_penalty*self.weights))
            # print progressing
            if verbose == True:
                sys.stdout.write("\rProgress: {:2.1f}".format(100 * iter/float(iteration)) \
                                    + "% ... Cost: " + str(cost))
                sys.stdout.flush()
            
    def l2_penalty_tuning(self, train_X, train_y, valid_X, valid_y, l2_penalty):
        # uses self.iteration, self.learning.
        lowest_cost = None
        best_l2_penalty = None
        print("Tuning Penalty...")
        for index, penalty in enumerate(l2_penalty_values):
            # train the model with training data
            self.fit(train_X, train_y, l2_penalty = penalty, learning_rate=self.learning_rate, iteration=self.iteration, verbose=False)
            # calculate the cost with valid data 
            y_pred = self.predict(valid_X)
            cost = np.sum((valid_y - y_pred)**2)
            if (best_l2_penalty == None or cost < lowest_cost):
                lowest_cost = cost
                best_l2_penalty = penalty
            print("[%d/%d] Penalty: %.5f    Cost: %.5e" %(index, len(l2_penalty), penalty, cost))
        print ("----------------")
        return [lowest_cost, best_l2_penalty]
    
    def r2_score(self, X, y):
        y_pred = self.predict(X)
        SSE = np.sum((y - y_pred) ** 2)
        SSTO = np.sum((y - y.mean()) ** 2)
        return (1 - (SSE / float(SSTO)))
```


```python
ridge_model = RidgeRegression()
ridge_model.fit(train_X, train_y, learning_rate=3e-14, l2_penalty=10, iteration=1e3, verbose=True)
```

    Progress: 99.9% ... Cost: 9.45856142325e+14


```python
print (ridge_model.weights)
```

    [  1.97537846e-01   1.38965059e-01   1.46934869e+02  -1.38621589e-01
       8.98123656e-02   3.28284864e-03   2.00213733e-01   1.17651520e+02
       2.92833489e+01   3.16686836e+00   2.85265616e+00]



```python
plt.plot(ridge_model.cost_history, label="Training cost")
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Cost', fontsize=14)
```




    <matplotlib.text.Text at 0x116b992b0>




![png](Ridge%20Regression%28Gradient%20Descent%29_files/Ridge%20Regression%28Gradient%20Descent%29_12_1.png)



```python
plt.scatter(train_X['sqft_living'], train_y)
plt.scatter(train_X['sqft_living'], ridge_model.predict(train_X))
plt.xlabel('sqft_living', fontsize=12)
plt.ylabel('price', fontsize=12)
print ("R2_score : ", ridge_model.r2_score(train_X, train_y))
```

    R2_score :  0.46699065492667735



![png](Ridge%20Regression%28Gradient%20Descent%29_files/Ridge%20Regression%28Gradient%20Descent%29_13_1.png)



```python
l2_penalty_values = np.logspace(-4, 4, num=5)
[lowest_cost, best_penalty] = ridge_model.l2_penalty_tuning(train_X, train_y, valid_X, valid_y, l2_penalty = l2_penalty_values)

print("Best Penalty : %.5f   Cost : %.5e " %(best_penalty, lowest_cost))
```

    Tuning Penalty...
    [0/5] Penalty: 0.00010    Cost: 2.62383e+14
    [1/5] Penalty: 0.01000    Cost: 2.62383e+14
    [2/5] Penalty: 1.00000    Cost: 2.62383e+14
    [3/5] Penalty: 100.00000    Cost: 2.62383e+14
    [4/5] Penalty: 10000.00000    Cost: 2.62383e+14
    ----------------
    Best Penalty : 0.00010   Cost : 2.62383e+14 


## Model with best L2 penalty


```python
best_model = RidgeRegression()
best_model.fit(train_X, train_y, l2_penalty=best_penalty, learning_rate=3.5e-14, iteration=5e3, verbose=True)
```

    Progress: 100.0% ... Cost: 9.17579502699e+14


```python
plt.scatter(test_X['sqft_living'], test_y)
plt.scatter(test_X['sqft_living'], best_model.predict(test_X))
plt.xlabel('sqft_living', fontsize=12)
plt.ylabel('price', fontsize=12)
best_model.r2_score(test_X, test_y)
```




    0.497580674551224




![png](Ridge%20Regression%28Gradient%20Descent%29_files/Ridge%20Regression%28Gradient%20Descent%29_17_1.png)


# LASSO regression coordinate descent

## Normalize features
In the house dataset, features vary wildly in their relative magnitude: `sqft_living` is very large overall compared to `bedrooms`, for instance. As a result, weight for `sqft_living` would be much smaller than weight for `bedrooms`. This is problematic because "small" weights are dropped first as `l1_penalty` goes up. 

To give equal considerations for all features, we need to **normalize features**. we divide each feature by its 2-norm so that the transformed feature has norm 1.


```python
# split the data set
[train, test] = train_test_split(house_price, test_size= 0.2)
[train, valid] = train_test_split(train, test_size= 0.2)
print('Train :', train.shape, '\nValid: ', valid.shape, '\nTest :', test.shape)
```

    Train : (13832, 12) 
    Valid:  (3458, 12) 
    Test : (4323, 12)



```python
# Splitting data
train_X = train.loc[:, train.columns != 'price']
train_y = train['price']
valid_X = valid.loc[:, valid.columns != 'price']
valid_y = valid['price']
test_X = test.loc[:, test.columns != 'price']
test_y = test['price']
print('X: ', train_X.shape, '\ny: ', train_y.shape)
```

    X:  (13832, 11) 
    y:  (13832,)



```python
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix / norms
    return(normalized_features, norms)
```


```python
# normalize data
[train_X, train_norms] = normalize_features(train_X)
print('Norms : ', *train_norms, sep='\n')
train_X.head()
```

    Norms : 
    411.43650786
    264.471524932
    267565.96541
    5117564.97463
    186.512063953
    10.0995049384
    407.953428715
    232098.51959
    62263.6082957
    6528.1934714
    6260.28026529





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>age</th>
      <th>age_renovated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10184</th>
      <td>0.009722</td>
      <td>0.009453</td>
      <td>0.011362</td>
      <td>0.006878</td>
      <td>0.010723</td>
      <td>0.0</td>
      <td>0.009805</td>
      <td>0.013098</td>
      <td>0.000000</td>
      <td>0.004749</td>
      <td>0.004952</td>
    </tr>
    <tr>
      <th>4741</th>
      <td>0.007292</td>
      <td>0.008508</td>
      <td>0.010390</td>
      <td>0.006157</td>
      <td>0.010723</td>
      <td>0.0</td>
      <td>0.007354</td>
      <td>0.011978</td>
      <td>0.000000</td>
      <td>0.005974</td>
      <td>0.006230</td>
    </tr>
    <tr>
      <th>14873</th>
      <td>0.007292</td>
      <td>0.008508</td>
      <td>0.004821</td>
      <td>0.000489</td>
      <td>0.010723</td>
      <td>0.0</td>
      <td>0.009805</td>
      <td>0.005558</td>
      <td>0.000000</td>
      <td>0.004749</td>
      <td>0.004952</td>
    </tr>
    <tr>
      <th>13030</th>
      <td>0.007292</td>
      <td>0.007562</td>
      <td>0.004672</td>
      <td>0.001688</td>
      <td>0.005362</td>
      <td>0.0</td>
      <td>0.012256</td>
      <td>0.005386</td>
      <td>0.000000</td>
      <td>0.007659</td>
      <td>0.007987</td>
    </tr>
    <tr>
      <th>7400</th>
      <td>0.007292</td>
      <td>0.003781</td>
      <td>0.003887</td>
      <td>0.001738</td>
      <td>0.005362</td>
      <td>0.0</td>
      <td>0.009805</td>
      <td>0.003447</td>
      <td>0.003855</td>
      <td>0.009191</td>
      <td>0.009584</td>
    </tr>
  </tbody>
</table>
</div>



# Implementing Coordinate Descent with normalized features
We seek to obtain a sparse set of weights by minimizing the LASSO cost function
```
SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|).
```
(By convention, we do not include `w[0]` in the L1 penalty term. We never want to push the intercept to zero.)

The absolute value sign makes the cost function non-differentiable, so simple gradient descent is not viable (you would need to implement a method called subgradient descent). Instead, we will use **coordinate descent**: at each iteration, we will fix all weights but weight `i` and find the value of weight `i` that minimizes the objective. That is, we look for
```
argmin_{w[i]} [ SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|) ]
```
where all weights other than `w[i]` are held to be constant. We will optimize one `w[i]` at a time, circling through the weights multiple times.  
  1. Pick a coordinate `i`
  2. Compute `w[i]` that minimizes the cost function `SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|)`
  3. Repeat Steps 1 and 2 for all coordinates, multiple times
 


```python
class LassoRegression():
    def __init__(self):
        self.weights = None
        self.l1_penalty = None
        self.iteration = None
        self.tolerance = None

    def predict(self, feature_matrix):
        predictions = np.dot(feature_matrix, self.weights)
        return(predictions)

    def lasso_coordinate_descent_step(self, i, X, y):
        # compute prediction
        prediction = self.predict(X)
        # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
        ro_i = (X.iloc[:,i] * (y - prediction + self.weights[i]*X.iloc[:,i]) ).sum()
        if i == 0: # intercept -- do not regularize
            new_weight_i = ro_i 
        elif ro_i < -self.l1_penalty/2.:
            new_weight_i = ro_i + (self.l1_penalty/2)
        elif ro_i > self.l1_penalty/2.:
            new_weight_i = ro_i - (self.l1_penalty/2)
        else:
            new_weight_i = 0.

        return new_weight_i

    def fit(self, X, y, l1_penalty, tolerance=1e-1, verbose=False):
        self.l1_penalty = l1_penalty
        self.verbose = verbose
        self.tolerance = tolerance
        self.weights = np.zeros(X.shape[1])
        
        converge = True    
        print_index = 0
        iter = 0
        while(converge):
            max_change = 0
            iter += 1
            changes = []
            for i in range(len(self.weights)):
                old_weights_i = self.weights[i]
                self.weights[i] = self.lasso_coordinate_descent_step(i, X, y)
                #print "new weight = %d" %weights[i]
                this_change = self.weights[i] - old_weights_i
                changes.append(this_change)
                max_change =  max(np.absolute(changes))
                print_index += 1
                if(verbose == True and print_index % 500 == 0):    
                    print("max change : %.3f" %(max_change) )
                            
            if (max_change < self.tolerance or iter > 1e3) :
                converge = False
            
    def r2_score(self, X, y):
        y_pred = self.predict(X)
        SSE = np.sum((y - y_pred) ** 2)
        SSTO = np.sum((y - y.mean()) ** 2)
        return (1 - (SSE / float(SSTO)))
    
    def l1_penalty_tuning(self, train_X, train_y, valid_X, valid_y, l1_penalty, tolerance=10):
        lowest_cost = None
        best_l1_penalty = None
        print("Tuning Penalty...")
        for index, penalty in enumerate(l1_penalty_values):
            self.fit(train_X, train_y, l1_penalty = penalty, tolerance=tolerance, verbose=False)
            cost = sum((valid_y-self.predict(valid_X))**2)
            if (best_l1_penalty == None or cost < lowest_cost):
                lowest_cost = cost
                best_l1_penalty = penalty
            print("[%d/%d] Penalty: %.5f    Cost: %.5e" %(index, len(l1_penalty), penalty, cost))
        print ("----------------")
        return [lowest_cost, best_l1_penalty]
```


```python
lasso_model = LassoRegression()
lasso_model.fit(train_X, train_y, l1_penalty=1, tolerance=10, verbose=True)
```

    max change : 455442.772
    max change : 168035.924
    max change : 57189.214
    max change : 19382.170
    max change : 6373.036
    max change : 2142.902
    max change : 531.528
    max change : 236.191
    max change : 6.130
    max change : 26.068



```python
plt.scatter(train_X['sqft_living'], train_y)
plt.scatter(train_X['sqft_living'], lasso_model.predict(train_X))
plt.xlabel('sqft_living', fontsize=12)
plt.ylabel('price', fontsize=12)
```




    <matplotlib.text.Text at 0x114130898>




![png](Ridge%20Regression%28Gradient%20Descent%29_files/Ridge%20Regression%28Gradient%20Descent%29_27_1.png)



```python
# since we normalized train data, we have to normalize test data as well.
normalized_valid_X = valid_X / train_norms
```


```python
# since train data is normalized, normalize the result weight to use it for test data set.
l1_penalty_values = np.logspace(1, 5, num=5)
[lowest_cost, best_penalty] = lasso_model.l1_penalty_tuning(train_X, train_y, normalized_valid_X, valid_y, l1_penalty = l1_penalty_values, tolerance=1e3)

print("Best Penalty : %.3f   Cost : %.5e " %(best_penalty, lowest_cost))
```

    Tuning Penalty...
    [0/5] Penalty: 10.00000    Cost: 2.08539e+14
    [1/5] Penalty: 100.00000    Cost: 2.08539e+14
    [2/5] Penalty: 1000.00000    Cost: 2.08538e+14
    [3/5] Penalty: 10000.00000    Cost: 2.08535e+14
    [4/5] Penalty: 100000.00000    Cost: 2.08560e+14
    ----------------
    Best Penalty : 10000.000   Cost : 2.08535e+14 



```python
best_lasso_model = LassoRegression()
best_lasso_model.fit(train_X, train_y, l1_penalty=best_penalty, tolerance=10, verbose=True)
```

    max change : 457412.689
    max change : 169163.286
    max change : 56956.610
    max change : 19608.141
    max change : 6816.760
    max change : 2659.781
    max change : 522.241
    max change : 786.343
    max change : 6.022
    max change : 579.898
    max change : 562.927
    max change : 557.157
    max change : 555.287
    max change : 554.651
    max change : 554.445
    max change : 554.375
    max change : 554.353
    max change : 0.003
    max change : 554.343
    max change : 0.000
    max change : 554.341
    max change : 554.341



```python
plt.scatter(test_X['sqft_living'], test_y)
plt.scatter(test_X['sqft_living'], best_lasso_model.predict(test_X/train_norms))
plt.xlabel('sqft_living', fontsize=12)
plt.ylabel('price', fontsize=12)
```




    <matplotlib.text.Text at 0x114190400>




![png](Ridge%20Regression%28Gradient%20Descent%29_files/Ridge%20Regression%28Gradient%20Descent%29_31_1.png)

