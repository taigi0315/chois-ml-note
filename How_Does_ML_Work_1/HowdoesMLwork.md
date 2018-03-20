
# How does ML work?
## what is the basic idea of machine learning


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```


```python
# load, and quick check data
house_price = pd.read_csv('data/house_price_train.csv')
house_price = house_price[['LotArea', 'SalePrice']]

plt.plot(house_price['LotArea'], house_price['SalePrice'], '.')
```




    [<matplotlib.lines.Line2D at 0x29671f039b0>]




![png](HowdoesMLwork_files/HowdoesMLwork_2_1.png)



```python
# To make it easier to understand, I pull out 15 random houses,
# we will use these house only for this post
sample_data = house_price.sample(15, random_state=3)
plt.plot(sample_data['LotArea'], sample_data['SalePrice'], '.')
plt.xlabel('Area', fontsize=12, color='blue')
plt.ylabel('Price', fontsize=12, color='blue')
```




    <matplotlib.text.Text at 0x296722e4128>




![png](HowdoesMLwork_files/HowdoesMLwork_3_1.png)



```python
# Simple function to draw lien
def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)
```

## 3 lines represent 3 different prediction models
as we can see, green is the best fit line out of 3. we can tell green line fits best by instict, but how did we know? why not red line or even yellow?


```python
# Check out our naive prediction line
plt.plot(sample_data['LotArea'], sample_data['SalePrice'], '.', markersize=8, label='_nolegend_')
plt.xlabel('Area', fontsize=12, color='blue')
plt.ylabel('Price', fontsize=12, color='blue')
graph('5*x', range(0, 20000))
graph('20*x', range(0, 20000))
graph('30*x', range(0, 20000))
plt.legend(['5', '20', '30'])
```




    <matplotlib.legend.Legend at 0x29672beec50>




![png](HowdoesMLwork_files/HowdoesMLwork_6_1.png)


---
# 머신러닝의 원리
만약 우리가 'W' 값을 알고있다면, 우리는 각각의 'x'값에 대한 'y' 값을 예측할 수 있습니다.
문제는 곱셈의 역함수가 없다는 것인데요.(곱셈의 역함수를 구해내는 것은 무척이나 까다로워 쓸 수 없습니다.) 
이 문제를 해결하기 위해 우리는 좌측의 'y'값을 우측으로 넘긴 후 'W' 값을 바꿔가며 가장 최소의 결과('Error')를 얻는 'W'값을 찾아내는 것입니다.

---
# What we are trying to do in Machine Learning
If we know the 'W' we can predict 'y' value when we have 'X' value.
problem is multiplication does not have inverse, so we move y value to the right side
and keep the left side as 0. Now we find 'W' that make smallest 'Error'

<img src="assets/Fig_1.png", width=400, height=400>
<img src="assets/Fig_2.png", width=400, height=400>

---
## See actual example
plot shows sum of errors on 15 house prices for each 'W' from 0 to 50
as you can see error starts from around 200k, hits zero, and goes down to -200k


```python
# Check the error based on W
plt.xlabel('W', fontsize=12)
plt.ylabel('Error', fontsize=12)

for w in range(0, 50):
    predict = w * sample_data['LotArea']
    error = np.mean(sample_data['SalePrice'] - predict)
    plt.scatter(w, error)
```


![png](HowdoesMLwork_files/HowdoesMLwork_11_0.png)


## Negative Error?
Negative error does not make sense, it happens when our predict house price was bigger than actual price. to solve this problem, simply we square the error, aka MSE(Mean Squared Error)


```python
# Check the squared error based on W
plt.xlabel('W', fontsize=12)
plt.ylabel('Error', fontsize=12)

for w in range(0, 50):
    predict = w * sample_data['LotArea']
    error = np.mean((sample_data['SalePrice'] - predict)**2)
    plt.scatter(w, error)
```


![png](HowdoesMLwork_files/HowdoesMLwork_13_0.png)

