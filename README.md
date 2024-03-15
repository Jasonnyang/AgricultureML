# AgricultureML

## Milestone 2
All of our data is numerical. In our dataset, we discovered that there were missing data values but they were encoded with the number -1. We could resolve this by using strategies like taking a random value in the range or by using the mean or we will throw those values out. In our case, we decided to throw the data values out because we were worried that when training our classifier, choosing a random value could cause the model to perform poorly since it will be trained on randomly generated data that may or may not conform to the trends in the data. If there was categorical data, we would need to label or one hot encode the values but the dataset is all numerical. 

Some strategies we will use to preprocess our data is doing min-max normalization in order to get all our numbers within a specific range. We will do this because comparing crop yields across different states is challenging since one crop could produce 50kg per hectare and others can produce 5 tons per hectare and it would be hard to compare without some sort of normalization technique. We will also throw out any missing values by only taking into account observations with values >= 0 because we don't want to normalize a distribution with numbers from -1 to n because it would throw the entire distribution off. We also ignored many of the columns in our dataset and only focused on the columns with yield and the corresponding state code because 42 columns of data were impractical to plot and analyze. 

## Milestone 3
The model is either underfitting or might be a good fit based on the train and test error. The train and test accuracy is very similar which is a good thing because it indicates that the model is not overfitting to the training data and causing it to perform poorly on the test dataset. However, logistic regression is a pretty simple model and because we have so many features in our dataset, it could not be capturing the complexities in the dataset as well as it could be, despite a high 82% accuracy on our model. The decision boundary might not be a good fit for the data because it is too simple to capture all the classes in the feature space.

The next two models we are thinking about trying is a deep neural network and a random forest. We have been learning about deep neural networks in class, and I think this will be more effective than logisitic regression in our classification problem. A DNN might be able to capture some of the hidden patterns in our data that logistic regression cannot capture. A random forest classifier may also be appropiate for our problem. Random forests also are used for classification and are also not as computationally expensive as a DNN.

The conclusion of our first model implies that there are some features of a state's crop yields and statistics which can be used to identify it. This is very good news because it means our problem is actually solvable. If there was no relationship between our X and y, we would have to choose a different set of features to look at. In order to improve our model, we could do some hyperparameter tuning. We could experiment with different solvers and/or stopping criteria for our Logistic Regression Model.

## Milestone 4
The data seemed sufficient because we have over ten thousand observations and labels to where we didn't need to generate additional data. We trained our second model which was a random forest classifier and we got Train Error: 0.01 and Test Error: 2.05 which was significantly smaller than the logistic regression model that we used initially. I think our model is a lot more complex than the logistic regression model and is closer to fitting just right compared to the logistic regression which was likely underfitting the data. We implemented hyper parameter tuning and k-fold cross validation so we can average out the model to give us an indicator if the model is overfitting. 

The result of the hyperparameter tuning is that a max depth of 20, minimum samples split of 2, and n estimators of 300 gave us the best results and got us to 96% accuracy. When we did k-fold cross validation for k=5 we got the following accuracies 0.77956792 0.91724643 0.86671549 0.83376053 0.81282051 and it gave us a Mean cross-validation score of 0.84.

The last model we are thinking about trying is a convolutional neural network. We have been learning about convolutional neural networks in class, and I think this will be more effective than logistic regression in our classification problem. A DNN might be able to capture some of the hidden patterns in our data that logistic regression cannot capture and it might be computationally less expensive than a super expansive decision tree. 

The conclusion of our second model implies that there are some features of a state's crop yields and statistics which can be used to identify it. It shows that there is a stronger link between the features and the state code than we originally imagined. In the end we had an accuracy of 84% by running the k-fold cross validation on the most optimal hyperparameters which was higher than the logistic regression likely because the model was more complex and was able to fit the data in a more complex manner. To improve the decision trees, it would be nice if we had the computational power to build more expansive trees and have a greater depth. 

## Milestone 5
We used the same data labels and scaling as in the Random Forest Classifier. For our loss funciton, we used sparse categorical cross entropy and experimented with different model optimizers (Adam and SGD). We have plenty of data points for each region, so we should have enough data poitns to achieve reasonable accuracy.

Tbe result of the hyperparameter tuning is the best parameters is the number of hidden layers of 1, model units is 512, learning rate of 0.001 and achieved a best accurafcy of 76%. When verifying our result with k-fold cross validaition (k=5), we got the following the accuracy scores for each fold: 0.62980593 0.77187843 0.70157451 0.73855731 0.66630037, and it gave a mean cross-validation score of 0.70.

The conclusion of our third model is that a neural network is slightly overfitting our data. It is possible that our data is not as complex as we orginally thought, and a random forest / logistic regression is a better model for our data. Initially, we overfitting even more (91% train accuracy vs a 60% test) with a hard coded 3 hidden layers, so we hyperparameter tuned on hidden layers and found that only 1 layer produced the best result. To improve our model, we can test other properties, like acivation functions or loss functions which could produce better resutls.


Setup Instructions: 
Download and clone the Colab notebook. Any necessary imports or pip installs are included at the top of the notebook already. 

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LZSfZZS8Dh74Gep_nfKY4WyoMBdtk_At?usp=sharing)
