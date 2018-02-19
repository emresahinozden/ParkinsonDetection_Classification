# ML-Project

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

