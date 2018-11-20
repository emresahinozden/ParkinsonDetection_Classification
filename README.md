# Machine Learning Project

This study is done for the Machine Learning project in University of Genoa under supervision of Lorenzo Rosasco and Alessandro Verri.
It is coded in Python and the process is explained by in-line comments and the file "comments.txt".


The study aims to create a model to detect Parkinson by the key stroke data.
The dataset contains keystroke logs collected from over 200 subjects, with and without Parkinson's Disease.


For more information: https://www.kaggle.com/valkling/tappy-keystroke-data-with-parkinsons-patients/home 



#Code explanation: 



The code is basically of 3 parts:

*data extraction, integration and cleaning

*running alogrithms

*visualization

In the first part, code extracts users txt files and users' input txt files and merge all in a dataframe
Then code cleans the data (removes missing values, irrelevant columns etc.)
After cleaning code deals with the categorical variables and numeralize them.
After numeralization, scaling is done.
And finally the dataset is split into training and test datasets.

In the second part, I assign models to training set and after fitting the model I predict test set by using the model
These part contains the following algorithms:
-Logistic Regression
-KNN
-SVM Kernel
-Naive Bayes
-Decision Tree
-Random Forest
-Ensemble Learning Voting (via KNN, SVC Kernel, Decision Tree models)

The final part is on visualization of the models and their predictions
I use a dataset of 1000 points randomly selected, 
I use another dataset because my test set is to big to visualize (in cosmetic means) 

I had used comment lines for clearer description and these lines are also in a text file called Comments.

