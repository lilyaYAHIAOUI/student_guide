# student_guide
In this work, we propose a solution to guide the baccalaureate students when filling in the wish list.  


# challenges :
The main challenge faced was dealing with the imbalanced dataset and how to encode rare categories in categorical data. 

# Solution :
After many experiments, We found that the best encoding is "entity embedding."

# Results:

metric              | Training           | validation           | test  |
--------------------| :-----------------:|:--------------------:| -----:|
 Loss               | 0.5837             | 0.62352              |0.6355 |
 Accuracy           | 0.8117             | 0.8107               |0.8037 |
 f1-score macro     | 0.7264             | 0.6168               |0.6060 |
 f1-score weighted  |0.8083              |0.8038                |0.7967 |

