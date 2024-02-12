# AgricultureML

All of our data is numerical. In our dataset, we discovered that there were missing data values but they were encoded with the number -1. We could resolve this by using strategies like taking a random value in the range or by using the mean or we will throw those values out. In our case, we decided to throw the data values out because we were worried that when training our classifier, choosing a random value could cause the model to perform poorly since it will be trained on randomly generated data that may or may not conform to the trends in the data. If there was categorical data, we would need to label or one hot encode the values but the dataset is all numerical. 

Some strategies we will use to preprocess our data is doing min-max normalization in order to get all our numbers within a specific range. We will do this because comparing crop yields across different states is challenging since one crop could produce 50kg per hectare and others can produce 5 tons per hectare and it would be hard to compare without some sort of normalization technique. We will also throw out any missing values by only taking into account observations with values >= 0 because we don't want to normalize a distribution with numbers from -1 to n because it would throw the entire distribution off. We also ignored many of the columns in our dataset and only focused on the columns with yield and the corresponding state code because 42 columns of data were impractical to plot and analyze. 

Setup Instructions: 
Download and clone the Colab notebook. Any necessary imports or pip installs are included at the top of the notebook already. 

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LZSfZZS8Dh74Gep_nfKY4WyoMBdtk_At?usp=sharing)
