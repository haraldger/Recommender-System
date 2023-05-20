Approach

System

Hyperparameter Tuning

A grid search was performed over the three relevant hyperparameters - number of epochs, learning rate and regularization. The outcome showed little variance among the choices, with final Root Mean Square Error ranging from 1.3 to approximately 1.35. The sole exception to this was the number of epochs, which is to be expected, where more epochs typically resulted in a lower error. However, 100 training epochs sometimes resulted in worse performance than 50, likely due to overfitting.

Limitations

User guide

Expected run-times