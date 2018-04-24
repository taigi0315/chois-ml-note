
# Implementing Logistic Classifier & Test the model on Amazon Product Review data set


```python
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from math import sqrt
import json
import matplotlib.pyplot as plt

%matplotlib inline
```


```python
# pd.read_csv intelligently converts input to python datatypes.
products = pd.read_csv("amazon_baby_subset.csv")
products = products.astype(str)
print ('Shape : ', products.shape)
```

    Shape :  (53072, 4)



```python
# Change format of feature
products['rating'] = products['rating'].astype(int)
products['sentiment'] = products['sentiment'].astype(int)
# fill in N/A's in the review column
products = products.fillna({'reveiw':''}) 
```


```python
# Write a function remove_punctuation that takes a line of text and removes all punctuation from that text
def remove_punctuation(text):
    import string
    return text.translate(string.punctuation)

products['review_clean'] = products['review'].apply(remove_punctuation)
products.head()
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
      <th>name</th>
      <th>review</th>
      <th>rating</th>
      <th>sentiment</th>
      <th>review_clean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Stop Pacifier Sucking without tears with Thumb...</td>
      <td>All of my kids have cried non-stop when I trie...</td>
      <td>5</td>
      <td>1</td>
      <td>All of my kids have cried non-stop when I trie...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nature's Lullabies Second Year Sticker Calendar</td>
      <td>We wanted to get something to keep track of ou...</td>
      <td>5</td>
      <td>1</td>
      <td>We wanted to get something to keep track of ou...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nature's Lullabies Second Year Sticker Calendar</td>
      <td>My daughter had her 1st baby over a year ago. ...</td>
      <td>5</td>
      <td>1</td>
      <td>My daughter had her 1st baby over a year ago. ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lamaze Peekaboo, I Love You</td>
      <td>One of baby's first and favorite books, and it...</td>
      <td>4</td>
      <td>1</td>
      <td>One of baby's first and favorite books, and it...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SoftPlay Peek-A-Boo Where's Elmo A Children's ...</td>
      <td>Very cute interactive book! My son loves this ...</td>
      <td>5</td>
      <td>1</td>
      <td>Very cute interactive book! My son loves this ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# read "important_words.json" file
with open('important_words.json') as data_file:
    important_words = json.load(data_file)

print('Number of importatnt words : ', len(important_words))
```

    Number of importatnt words :  193



```python
# now we proceed with the second item. For each word in important_words, 
# we compute a count for the number of times the word occurs in the review.
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s: s.split().count(word))
```


```python
# Split train/valid data set
train_data = products.sample(frac=0.8)
validation_data = products.drop(train_data.index)
```


```python
def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    features_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()
    return(features_matrix, label_array)
```


```python
feature_matrix_train, sentiment_train = get_numpy_data(train_data, important_words, 'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(validation_data, important_words, 'sentiment')

print ('Input feature(X) : ', feature_matrix_train.shape, 'Output(y) : ', sentiment_train.shape)
```

    Input feature(X) :  (42458, 194) Output(y) :  (42458,)


## Building a logistic regression

$$\ell\ell(\mathbf{w}) = \sum_{i=1}^N \Big( (\mathbf{1}[y_i = +1] - 1)\mathbf{w}^T h(\mathbf{x}_i) - \ln\left(1 + \exp(-\mathbf{w}^T h(\mathbf{x}_i))\right) \Big) \color{red}{-\lambda\|\mathbf{w}\|_2^2} $$


```python
class logistic_classifier():
    def __init__(self):
        self.coefficients = np.zeros(1)
        self.l2_penalty = 0;
        self.iteration = 501
        self.learning_rate = 0
    
    def predict_probability(self, feature_matrix):
        #Take dot product of feature_matrix and coefficients
        score = np.dot(feature_matrix, self.coefficients)
        #Compute P(y_i = +1|x_i, w) using the link function
        predictions = 1.0/(1 + np.exp(-score))
        return predictions
    
    #Compute derivative of log likelihood with respect to a single coefficient
    def feature_derivative_with_L2(self, errors, feature, coefficient, feature_is_constant):
        #Compute the dot product of errors and feature(without L2 penalty)
        derivative = np.dot(errors, feature)
        
        #add L2 penalty term for any feature that isn't the intercept
        if not feature_is_constant:
            derivative -= 2 * self.l2_penalty * coefficient
        return derivative
    
    def compute_log_likelihood_with_L2(self, feature_matrix, sentiment):
        indicator = (sentiment == +1)
        scores = np.dot(feature_matrix, self.coefficients)
        lp = np.sum((indicator-1) * scores - np.log(1. + np.exp(-scores))) - self.l2_penalty*np.sum(self.coefficients[1:]**2)
        return lp
    
    def fit(self, feature_matrix, sentiment, learning_rate, l2_penalty, iteration):
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.iteration = iteration
        self.coefficients = np.zeros(feature_matrix.shape[1])
        print (self.l2_penalty, self.iteration, self.learning_rate)
        for itr in range(iteration):
            #Predict P(y_i = +1|x_1,w) using your predict_probability() function
            predictions = self.predict_probability(feature_matrix)

            #compute indicator value for (y_i = +1)
            indicator = (sentiment==+1)

            #Compute the errors as indicator - predictions
            errors = indicator - predictions
        
            for j in range(len(self.coefficients)): #loop over each coefficient
                is_intercept = (j==0)
                #Recall that feature_matrix[:,j] is the feature column associated with coefficients[j]
                #compute the derivative for coefficients[j]. Save it in a variable called derivative
                derivative = self.feature_derivative_with_L2(errors, feature_matrix[:,j], self.coefficients[j], is_intercept)
                #add step size times the derivative to the current coefficient(l2_penalty is already added)
                self.coefficients[j] += learning_rate * derivative

            #Checking whether log likelihood is increasing
            if (itr <= 100 and itr %10 ==0) or \
                (itr <= 1000 and itr %100 ==0) or (itr <= 10000 and itr %1000 ==0) or itr % 10000 ==0:
                    lp = self.compute_log_likelihood_with_L2(feature_matrix, sentiment)
                    print ('iteration %*d : log likelihood of observed labels = %.8f' % \
                    (int(np.ceil(np.log10(iteration ))), itr, lp))
       
    def get_accuracy(self, feature_matrix, sentiment):
        #compute scores using feature_matrix, coefficients
        scores = np.dot(feature_matrix, self.coefficients)
        #threshold scores by 0
        positive = scores > 0
        negative = scores <= 0
        scores[positive] = 1
        scores[negative] = -1

        correct = float((scores == sentiment).sum())
        total = float(len(sentiment))
        accuracy = float(correct / total)
        return accuracy
```


```python
learning_rate = 5e-6
iteration = 501
```


```python
l2_penalty = 0
model_0_penalty = logistic_classifier()
model_0_penalty.fit(feature_matrix_train, sentiment_train, learning_rate, l2_penalty, iteration)
```

    0 501 5e-06
    iteration   0 : log likelihood of observed labels = -29285.81858112
    iteration  10 : log likelihood of observed labels = -28080.97525573
    iteration  20 : log likelihood of observed labels = -27166.78266349
    iteration  30 : log likelihood of observed labels = -26446.80354887
    iteration  40 : log likelihood of observed labels = -25864.03157757
    iteration  50 : log likelihood of observed labels = -25381.53400935
    iteration  60 : log likelihood of observed labels = -24974.54769449
    iteration  70 : log likelihood of observed labels = -24625.91505221
    iteration  80 : log likelihood of observed labels = -24323.38617617
    iteration  90 : log likelihood of observed labels = -24057.97892510
    iteration 100 : log likelihood of observed labels = -23822.95109833
    iteration 200 : log likelihood of observed labels = -22396.11470688
    iteration 300 : log likelihood of observed labels = -21703.52663790
    iteration 400 : log likelihood of observed labels = -21287.34588578
    iteration 500 : log likelihood of observed labels = -21008.20079900



```python
l2_penalty = 5
model_5_penalty = logistic_classifier()
model_5_penalty.fit(feature_matrix_train, sentiment_train, learning_rate, l2_penalty, iteration)
```

    5 501 5e-06
    iteration   0 : log likelihood of observed labels = -29285.82221230
    iteration  10 : log likelihood of observed labels = -28081.65128806
    iteration  20 : log likelihood of observed labels = -27168.90179711
    iteration  30 : log likelihood of observed labels = -26450.80057875
    iteration  40 : log likelihood of observed labels = -25870.15723019
    iteration  50 : log likelihood of observed labels = -25389.93326703
    iteration  60 : log likelihood of observed labels = -24985.30188410
    iteration  70 : log likelihood of observed labels = -24639.06564554
    iteration  80 : log likelihood of observed labels = -24338.94890382
    iteration  90 : log likelihood of observed labels = -24075.95250119
    iteration 100 : log likelihood of observed labels = -23843.32280950
    iteration 200 : log likelihood of observed labels = -22438.87115969
    iteration 300 : log likelihood of observed labels = -21765.45838523
    iteration 400 : log likelihood of observed labels = -21365.80392894
    iteration 500 : log likelihood of observed labels = -21101.09703273



```python
l2_penalty = 10
model_10_penalty = logistic_classifier()
model_10_penalty.fit(feature_matrix_train, sentiment_train, learning_rate, l2_penalty, iteration)
```

    10 501 5e-06
    iteration   0 : log likelihood of observed labels = -29285.82584348
    iteration  10 : log likelihood of observed labels = -28082.32687618
    iteration  20 : log likelihood of observed labels = -27171.01819921
    iteration  30 : log likelihood of observed labels = -26454.78999693
    iteration  40 : log likelihood of observed labels = -25876.26751800
    iteration  50 : log likelihood of observed labels = -25398.30645453
    iteration  60 : log likelihood of observed labels = -24996.01635852
    iteration  70 : log likelihood of observed labels = -24652.15999611
    iteration  80 : log likelihood of observed labels = -24354.43605735
    iteration  90 : log likelihood of observed labels = -24093.82845717
    iteration 100 : log likelihood of observed labels = -23863.57223308
    iteration 200 : log likelihood of observed labels = -22481.13389814
    iteration 300 : log likelihood of observed labels = -21826.34801524
    iteration 400 : log likelihood of observed labels = -21442.54546071
    iteration 500 : log likelihood of observed labels = -21191.51062622



```python
l2_penalty = 1e2
model_1e2_penalty = logistic_classifier()
model_1e2_penalty.fit(feature_matrix_train, sentiment_train, learning_rate, l2_penalty, iteration)
```

    100.0 501 5e-06
    iteration   0 : log likelihood of observed labels = -29285.89120473
    iteration  10 : log likelihood of observed labels = -28094.41185984
    iteration  20 : log likelihood of observed labels = -27208.65082589
    iteration  30 : log likelihood of observed labels = -26525.31666231
    iteration  40 : log likelihood of observed labels = -25983.67570644
    iteration  50 : log likelihood of observed labels = -25544.67258300
    iteration  60 : log likelihood of observed labels = -25182.28054461
    iteration  70 : log likelihood of observed labels = -24878.56251986
    iteration  80 : log likelihood of observed labels = -24620.77514263
    iteration  90 : log likelihood of observed labels = -24399.61885697
    iteration 100 : log likelihood of observed labels = -24208.14620146
    iteration 200 : log likelihood of observed labels = -23165.30666604
    iteration 300 : log likelihood of observed labels = -22768.32025229
    iteration 400 : log likelihood of observed labels = -22581.60671026
    iteration 500 : log likelihood of observed labels = -22483.70730446



```python
l2_penalty = 1e3
model_1e3_penalty = logistic_classifier()
model_1e3_penalty.fit(feature_matrix_train, sentiment_train, learning_rate, l2_penalty, iteration)
```

    1000.0 501 5e-06
    iteration   0 : log likelihood of observed labels = -29286.54481726
    iteration  10 : log likelihood of observed labels = -28207.74244233
    iteration  20 : log likelihood of observed labels = -27541.07895090
    iteration  30 : log likelihood of observed labels = -27114.25505568
    iteration  40 : log likelihood of observed labels = -26834.16487061
    iteration  50 : log likelihood of observed labels = -26646.66551516
    iteration  60 : log likelihood of observed labels = -26519.09206952
    iteration  70 : log likelihood of observed labels = -26431.11755232
    iteration  80 : log likelihood of observed labels = -26369.76214077
    iteration  90 : log likelihood of observed labels = -26326.55811852
    iteration 100 : log likelihood of observed labels = -26295.88171307
    iteration 200 : log likelihood of observed labels = -26218.93679001
    iteration 300 : log likelihood of observed labels = -26215.09090526
    iteration 400 : log likelihood of observed labels = -26214.84127150
    iteration 500 : log likelihood of observed labels = -26214.82258876



```python
l2_penalty = 1e5
model_1e5_penalty = logistic_classifier()
model_1e5_penalty.fit(feature_matrix_train, sentiment_train, learning_rate, l2_penalty, iteration)
```

    100000.0 501 5e-06
    iteration   0 : log likelihood of observed labels = -29358.44219549
    iteration  10 : log likelihood of observed labels = -29358.35978606
    iteration  20 : log likelihood of observed labels = -29358.35566935
    iteration  30 : log likelihood of observed labels = -29358.35417612
    iteration  40 : log likelihood of observed labels = -29358.35363450
    iteration  50 : log likelihood of observed labels = -29358.35343804
    iteration  60 : log likelihood of observed labels = -29358.35336678
    iteration  70 : log likelihood of observed labels = -29358.35334093
    iteration  80 : log likelihood of observed labels = -29358.35333156
    iteration  90 : log likelihood of observed labels = -29358.35332816
    iteration 100 : log likelihood of observed labels = -29358.35332693
    iteration 200 : log likelihood of observed labels = -29358.35332622
    iteration 300 : log likelihood of observed labels = -29358.35332622
    iteration 400 : log likelihood of observed labels = -29358.35332622
    iteration 500 : log likelihood of observed labels = -29358.35332622


# Visualize the result


```python
#but we gonna use this DataFrame
table = pd.DataFrame({'word': important_words, 
                      'l2_penalty_0': model_0_penalty.coefficients[1:],
                      'l2_penalty_5': model_5_penalty.coefficients[1:],
                      'l2_penalty_10': model_10_penalty.coefficients[1:],
                      'l2_penalty_1e2': model_1e2_penalty.coefficients[1:],
                      'l2_penalty_1e3': model_1e3_penalty.coefficients[1:],
                      'l2_penalty_1e5': model_1e5_penalty.coefficients[1:]})
```


```python
table = table.sort_values(['l2_penalty_0'], ascending=[0])
table = table[['word', 'l2_penalty_0', 'l2_penalty_5', 'l2_penalty_10', 'l2_penalty_1e2', 'l2_penalty_1e3', 'l2_penalty_1e5']]
positive_words = table[1:6]['word']
negative_words = table[-6:-1]['word']
print ('Positive words : \n', positive_words)
print ('Negative words : \n', negative_words)
```

    Positive words : 
     3        love
    7        easy
    2       great
    33    perfect
    8      little
    Name: word, dtype: object
    Negative words : 
     99          thought
    168        returned
    96            money
    105    disappointed
    113          return
    Name: word, dtype: object



```python
table.head()
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
      <th>word</th>
      <th>l2_penalty_0</th>
      <th>l2_penalty_5</th>
      <th>l2_penalty_10</th>
      <th>l2_penalty_1e2</th>
      <th>l2_penalty_1e3</th>
      <th>l2_penalty_1e5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>loves</td>
      <td>1.109108</td>
      <td>1.097863</td>
      <td>1.086821</td>
      <td>0.917918</td>
      <td>0.360573</td>
      <td>0.006118</td>
    </tr>
    <tr>
      <th>3</th>
      <td>love</td>
      <td>1.095556</td>
      <td>1.085786</td>
      <td>1.076199</td>
      <td>0.930326</td>
      <td>0.433704</td>
      <td>0.009027</td>
    </tr>
    <tr>
      <th>7</th>
      <td>easy</td>
      <td>1.003390</td>
      <td>0.994685</td>
      <td>0.986143</td>
      <td>0.856038</td>
      <td>0.407102</td>
      <td>0.008468</td>
    </tr>
    <tr>
      <th>2</th>
      <td>great</td>
      <td>0.779377</td>
      <td>0.773183</td>
      <td>0.767100</td>
      <td>0.673842</td>
      <td>0.335499</td>
      <td>0.006996</td>
    </tr>
    <tr>
      <th>33</th>
      <td>perfect</td>
      <td>0.708485</td>
      <td>0.700817</td>
      <td>0.693282</td>
      <td>0.577465</td>
      <td>0.200807</td>
      <td>0.002981</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.rcParams['figure.figsize'] = 10, 6

def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')
    
    xx = l2_penalty_list
    plt.plot(xx, [0.]*len(xx), '--', lw=1, color='k')
    
    table_positive_words = table[table['word'].isin(positive_words)]
    table_negative_words = table[table['word'].isin(negative_words)]
    del table_positive_words['word']
    del table_negative_words['word']
    
    for i in range(len(positive_words)):
        color = cmap_positive(0.8*((i+1)/(len(positive_words)*1.2)+0.15))
        plt.plot(xx, table_positive_words[i:i+1].as_matrix().flatten(),
                 '-', label=positive_words.iloc[i], linewidth=4.0, color=color)
        
    for i in range(len(negative_words)):
        color = cmap_negative(0.8*((i+1)/(len(negative_words)*1.2)+0.15))
        plt.plot(xx, table_negative_words[i:i+1].as_matrix().flatten(),
                 '-', label=negative_words.iloc[i], linewidth=4.0, color=color)
        
    plt.legend(loc='best', ncol=3, prop={'size':16}, columnspacing=0.5)
    plt.axis([1, 1e5, -1, 2])
    plt.title('Coefficient path')
    plt.xlabel('L2 penalty ($\lambda$)')
    plt.ylabel('Coefficient value')
    plt.xscale('log')
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()


make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list=[0, 5, 10, 1e2, 1e3, 1e5])
```


![png](Logistic%20Classifier%20Scratch%20Implementation_files/Logistic%20Classifier%20Scratch%20Implementation_23_0.png)



```python
def get_accuracy(feature_matrix, coefficients, sentiment):
    #compute scores using feature_matrix, coefficients
    scores = np.dot(feature_matrix, coefficients)
    #threshold scores by 0
    positive = scores > 0
    negative = scores <= 0
    scores[positive] = 1
    scores[negative] = -1

    correct = float((scores == sentiment).sum())
    total = float(len(sentiment))
    accuracy = float(correct / total)
    return accuracy
```


```python
train_accuracy = {}
train_accuracy[0] = model_0_penalty.get_accuracy(feature_matrix_train, sentiment_train)
train_accuracy[5] = model_5_penalty.get_accuracy(feature_matrix_train, sentiment_train)
train_accuracy[10] = model_10_penalty.get_accuracy(feature_matrix_train, sentiment_train)
train_accuracy[1e2] = model_1e2_penalty.get_accuracy(feature_matrix_train, sentiment_train)
train_accuracy[1e3] = model_1e3_penalty.get_accuracy(feature_matrix_train, sentiment_train)
train_accuracy[1e5] = model_1e5_penalty.get_accuracy(feature_matrix_train, sentiment_train)
print (train_accuracy)
```

    {0: 0.7688303735456216, 5: 0.7687361627961751, 10: 0.7686419520467286, 100.0: 0.7677469499269867, 1000.0: 0.7568656083659145, 100000.0: 0.7286494889066842}



```python
validation_accuracy = {}
validation_accuracy[0] = model_0_penalty.get_accuracy(feature_matrix_valid, sentiment_valid)
validation_accuracy[5] = model_5_penalty.get_accuracy(feature_matrix_valid, sentiment_valid)
validation_accuracy[10] = model_10_penalty.get_accuracy(feature_matrix_valid, sentiment_valid)
validation_accuracy[1e2] = model_1e2_penalty.get_accuracy(feature_matrix_valid, sentiment_valid)
validation_accuracy[1e3] = model_1e3_penalty.get_accuracy(feature_matrix_valid, sentiment_valid)
validation_accuracy[1e5] = model_1e5_penalty.get_accuracy(feature_matrix_valid, sentiment_valid)
print (validation_accuracy)
```

    {0: 0.7652157527793481, 5: 0.7656868287167892, 10: 0.7654983983418127, 100.0: 0.7648388920293951, 1000.0: 0.7547578669681553, 100000.0: 0.7302619182212172}



```python
# Optional. Plot accuracy on training and validation sets over choice of L2 penalty.
plt.rcParams['figure.figsize'] = 10, 6

sorted_list = sorted(train_accuracy.items(), key=lambda x:x[0])
plt.plot([p[0] for p in sorted_list], [p[1] for p in sorted_list], 'bo-', linewidth=4, label='Training accuracy')
sorted_list = sorted(validation_accuracy.items(), key=lambda x:x[0])
plt.plot([p[0] for p in sorted_list], [p[1] for p in sorted_list], 'ro-', linewidth=4, label='Validation accuracy')
plt.xscale('symlog')
plt.axis([0, 1e5, 0.70, 0.78])
plt.legend(loc='lower left')
plt.rcParams.update({'font.size': 18})
plt.tight_layout
```




    <function matplotlib.pyplot.tight_layout>




![png](Logistic%20Classifier%20Scratch%20Implementation_files/Logistic%20Classifier%20Scratch%20Implementation_27_1.png)

