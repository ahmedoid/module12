## Apply Machine Learning Classification Models to Iris Flowers Dataset
> Ahmed Aljuaid
>
 #### A.Data Descriptive:
 ##### 1.Head of data:
 Print first few of daatset
 
|   | sepal.width | petal.length | petal.width variety | petal.width | variety |
|---|-------------|--------------|---------------------|-------------|---------|
| 0 | 5.1         | 3.5          | 1.4                 | 0.2         | Setosa  |
| 1 | 4.9         | 3.0          | 1.4                 | 0.2         | Setosa  |
| 2 | 4.7         | 3.2          | 1.3                 | 0.2         | Setosa  |
| 3 | 4.6         | 3.1          | 1.5                 | 0.2         | Setosa  |
| 4 | 5.0         | 3.6          | 1.4                 | 0.2         | Setosa  |

 ##### 2.Features and types:
RangeIndex: 150 entries, 0 to 149   
Data columns (total 5 columns):
 
| #   | Column       | Non-Null Count | Dtype   |
|-----|--------------|----------------|---------|
| --- | ------       | -------------- | -----   |
| 0   | sepal.length | 150 non-null   | float64 |
| 1   | sepal.width  | 150 non-null   | float64 |
| 2   | petal.length | 150 non-null   | float64 |
| 3   | petal.width  | 150 non-null   | float64 |
| 4   | variety      | 150 non-null   | object  |
dtypes: float64(4), object(1)  
memory usage: 6.0+ KB   
 ##### 3.Value count for each type :
 Virginica     50  
 Setosa        50  
 Versicolor    50  
 Name: variety, dtype: int64
 
 ![alt text](https://github.com/ahmedoid/module12/blob/master/myplot.png?raw=true)


 #### B.Label encoding
 Result after encoding:  
[0 1 2]   

#### C.Data Splitting [Test,Training]
hape of X-Training : (90, 4)   
Shape of Y-Training : (90,)   
Shape of X-Test : (60, 4)   
Shape of Y-Test : (60,)  
Length of list 25  
Max of list 1.0   

 ![alt text](https://github.com/ahmedoid/module12/blob/master/myplot1.png?raw=true)


##  K Nearest Neighbors
##### Classification Report
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 1.00      | 1.00   | 1.00     | 20      |
| 1            | 1.00      | 0.95   | 0.98     | 21      |
| 2            | 0.95      | 1.00   | 0.97     | 19      |
| accuracy     | 0.98      | 60     | float64  |         |
| macro avg    | 0.98      | 0.98   | 0.98     | 60      |
| weighted avg | 0.98      | 0.98   | 0.98     | 60      |

##### Confusion Matrix
 [[20  0  0]  
 [ 0 20  1]  
 [ 0  0 19]]  
 
 Accuracy Score0.9833333333333333
 
 ##  Random Forests
 ####Classification Report
 |              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 1.00      | 1.00   | 1.00     | 20      |
| 1            | 0.95      | 0.95   | 0.95     | 21      |
| 2            | 0.95      | 0.95   | 0.95     | 19      |
| accuracy     | 0.97      | 60     | float64  |         |
| macro avg    | 0.97      | 0.97   | 0.97     | 60      |
| weighted avg | 0.97      | 0.97   | 0.97     | 60      |

##### Confusion Matrix
 [[20  0  0]  
 [ 0 20  1]  
 [ 0  1 18]]
 
 Accuracy Score0.9666666666666667
