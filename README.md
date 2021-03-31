# Decision Tree

## 1. Importing Library
--------------------------------


```python
import pandas as pd
import numpy as np
import math
```

## 2. Preparing Data Using Pandas
---------------------------------------------


```python
data = pd.DataFrame({
    'customer_id':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    'gender':['M','M','M','M','M','M','F','F','F','F','M','M','M','M','F','F','F','F','F','F'],
    'car_type':['Family','Sports','Sports','Sports','Sports','Sports','Sports','Sports','Sports','Luxury','Family','Family','Family','Luxury','Luxury','Luxury','Luxury','Luxury','Luxury','Luxury'],
    'shirt_size':['Small','Medium','Medium','Large','Extra Large','Extra Large','Small','Small','Medium','Large','Large','Extra Large','Medium','Extra Large','Small','Small','Medium','Medium','Medium','Large'],
    'class':['0','0','0','0','0','0','0','0','0','0','0','1','1','1','0','1','1','1','1','1']
})
columns = data.columns.tolist()
columns.remove('customer_id')
data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>gender</th>
      <th>car_type</th>
      <th>shirt_size</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>M</td>
      <td>Family</td>
      <td>Small</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>M</td>
      <td>Sports</td>
      <td>Medium</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>M</td>
      <td>Sports</td>
      <td>Medium</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>M</td>
      <td>Sports</td>
      <td>Large</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>M</td>
      <td>Sports</td>
      <td>Extra Large</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>M</td>
      <td>Sports</td>
      <td>Extra Large</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>F</td>
      <td>Sports</td>
      <td>Small</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>F</td>
      <td>Sports</td>
      <td>Small</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>F</td>
      <td>Sports</td>
      <td>Medium</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>F</td>
      <td>Luxury</td>
      <td>Large</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>M</td>
      <td>Family</td>
      <td>Large</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>M</td>
      <td>Family</td>
      <td>Extra Large</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>M</td>
      <td>Family</td>
      <td>Medium</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>M</td>
      <td>Luxury</td>
      <td>Extra Large</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>F</td>
      <td>Luxury</td>
      <td>Small</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>F</td>
      <td>Luxury</td>
      <td>Small</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>F</td>
      <td>Luxury</td>
      <td>Medium</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>F</td>
      <td>Luxury</td>
      <td>Medium</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>F</td>
      <td>Luxury</td>
      <td>Medium</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>F</td>
      <td>Luxury</td>
      <td>Large</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Data Slicing


```python
data.groupby(['class','gender'], as_index=True)['gender'].count().reset_index(name='count')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>gender</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>F</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>M</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>F</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>M</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Impurity Function
--------------------------------

### GINI


```python
def gini(x):
    num_data = sum(x)
    sigma = 0
    for i in x:
        sigma += (i/num_data)**2
    return 1-sigma
```

### Entropy


```python
def entropy(x):
    num_data = sum(x)
    sigma = 0
    for i in x:
        sigma += (i/num_data)*math.log(i/num_data)
    return sigma*-1
```

### Missclassification Error


```python
def miss_error(x):
    num_data = sum(x)
    max_val = 0
    for i in x:
        max_val = max(max_val, (i/num_data))
    return 1-max_val
```

## 4. Calculate All Data Frame Impurity
---------------------------------------------------


```python
def impurity(dataset, cols, method=0):
    '''
    Method: 0 for gini (default), 1 for entropy, 2 for missclassification error
    '''
    impurity = dict()
    method_list = ['gini', 'entropy', 'miss_error']
    method = method_list[method]
    
    class_data = dataset.groupby(['class'], as_index=True)['class'].count().reset_index(name='count')
    impurity['class'] = globals()[method](list(class_data['count']))
    
    cols.remove('class')
    for col in cols:
        sliced = dataset.groupby(['class',col], as_index=True)[col].count().reset_index(name='count')
        num_data = sum(sliced['count'])
        unique_value = sliced[col].unique()
        tmp = dict()
        for val in unique_value:
            sliced_by_value = list(sliced[sliced[col]==val]['count'])
            tmp[val] = globals()[method](sliced_by_value)*(sum(sliced_by_value)/num_data)
        impurity[col] = tmp
    return impurity
```

### Test Function Using Dataset|


```python
from pprint import pprint
data_impurity = impurity(data, columns.copy())
pprint(data_impurity,width=1)
```

    {'car_type': {'Family': 0.1,
                  'Luxury': 0.15000000000000002,
                  'Sports': 0.0},
     'class': 0.48,
     'gender': {'F': 0.25,
                'M': 0.21000000000000002},
     'shirt_size': {'Extra Large': 0.1,
                    'Large': 0.07500000000000001,
                    'Medium': 0.17142857142857146,
                    'Small': 0.07999999999999996}}


## Training
---------------
### 1st node


```python
data_impurity_1st = data_impurity.copy()
class_impurity = data_impurity_1st['class']
gain = dict()
data_impurity_1st.pop('class')
for key in data_impurity_1st.keys():
    gain[key] = class_impurity - sum(data_impurity_1st[key].values())
pprint(gain)
min_gain = sorted(gain, key=gain.get, reverse=True)[0]
print('\nMinimal Information Gain: ', min_gain,'(',gain[min_gain],')')
```

    {'car_type': 0.22999999999999998,
     'gender': 0.019999999999999962,
     'shirt_size': 0.05357142857142855}
    
    Minimal Information Gain:  gender ( 0.019999999999999962 )



```python

```
