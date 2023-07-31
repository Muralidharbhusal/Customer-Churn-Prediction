
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string, re
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_curve, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

df =pd.read_csv("Customer-Churn-Records.csv")

# Looking into the data, there are several irrevelant columns in the data like RowNumber, 
#Surname and CustomerId which I think donot have nay impact on the outcme. So let's drop those columns.

df_new = df.drop(["RowNumber","CustomerId","Surname"], axis = 1)
df_new.head()

print("Null Values:\n\n", df_new.isnull().sum())
print("<==================================================> \n")
print("Duplicated Values: ", df_new.duplicated().sum())


# The dataset looks clean with no null values and no duplicate values making it easier for us to clean the data. Now, let's check other properties of the dataset.
# 
# ## Statistical Summary of the dataset

df_new.describe() # It provides a summary of the central tendency, dispersion, and shape of the distribution of numerical data in the DataFrame
df_new.info()
# Check how the distribution of the target values is.
# Here the target values are in Exited column

df_new['Exited'].value_counts()

# Exited is Our Target variable. Let's rename it to churn for convinience
df_new.rename(columns = {'Exited':'Churn'}, inplace = True)
df_new.head(5)


# The distribution looks very uneven. This does not directly affect the prediction or even the final result of the model. Let's see what we can do about it.
# Exploratory Data Analysis and Visualization
# 1. Univariate Analysis
# First step we will be looking at categorical data. Categorical data in the given dataset are:
# 
# 1. Geography
# 2. Gender
# 3. Tenure
# 4. NumOfProducts
# 5. HasCrCard
# 6. IsActiveMember
# 7. Churn
# 8. Complain
# 9. Satisfaction Score
# 10. Card Type

#
plt.figure(figsize = (8, 5))

plt.plot(5, 2, 1)
plt.gca().set_title('Variable Geography')
sns.countplot(x = 'Geography', palette = ["red", "green", "blue"], data = df_new)

# Pie chart
plt.pie(df_new["Gender"].value_counts(), labels=df["Gender"].value_counts().index, autopct='%1.1f%%', startangle=90)
plt.title("Variable Gender")
plt.show()

# Let'd draw similar pie chart in anther variable
# import random
counts = df_new["Tenure"].value_counts()
label = counts.index
plt.pie(counts, labels = label, startangle = 90, autopct = "%1.1f%%", shadow = True, explode = [0,0.2,0,0,0,0.1,0,0,0.1,0.2,0.1])
plt.axis('equal')
plt.title("Variable Tenure in Years")
plt.show()


# Number of Products
count = df_new["Card Type"].value_counts().values
names = df_new["Card Type"].value_counts().index
plt.figure(figsize=(8, 5))
plt.bar(names,df_new["Card Type"].value_counts(), label = names, color = ['white','gold','silver','lavender'])
for i, v in enumerate(count):
    plt.text(i, v, str(v), ha='center', va='bottom', color = 'white')
plt.xlabel('Card Type')
plt.ylabel('Counts')
plt.title("Variable Card Type")
plt.legend(prop = {'size':5}, loc = 'upper right')
plt.gca().set_facecolor('black')
plt.show()

# Variable no of products
count = df_new['NumOfProducts'].value_counts().values
value = df_new['NumOfProducts'].value_counts().index
# print(value, count)
plt.bar(value, count, color = ['green', 'silver', 'gold', 'blue'])
plt.ylabel("Count")
plt.xlabel("No of Products")
plt.title("Variable Num Of Products")
plt.show()

# Let's See what number of customers have credit cards anw what number don't
plt.figure(figsize = (8, 5))

plt.plot(5, 2, 1)
plt.gca().set_title('Variable HasCrCard')
sns.countplot(x = 'HasCrCard', palette = ["red", "green"], data = df_new)


# Active member or not
plt.figure(figsize = (8, 5))

plt.plot(5, 2, 1)
plt.gca().set_title('Variable Is Active Member')
sns.countplot(x = 'IsActiveMember', palette = ["red", "green"], data = df_new)

# Variable Satisfaction Score
# plt.figure(figsize = (8,5))
data = df_new['Satisfaction Score'].value_counts().values
score = df_new['Satisfaction Score'].value_counts().index
plt.pie(data, labels = score, startangle=90, autopct="%1.1f%%", explode = [0.1,0.1,0.1,0.1,0.1], shadow = True)
plt.title("Satisfaction Score")
plt.show()


# ## Continuous Data
# 
# Let's take a look at the distribution of the continuous data in the dataframe.


# Plotting CreditScore

def plot_continuous(a):
    print(a)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(df[a], kde=True, ax=axes[0])
    axes[0].set_title(f'{a} Histogram')

    sns.boxplot(df[a],ax=axes[1])
    axes[1].set_title(f'{a} Boxplot')

    return plt.show()
# plot_continuous()


plot_continuous("CreditScore")

# Plotting Age
plot_continuous('Age')

# Point Earned
plot_continuous('Point Earned')

# Estimated Salary
plot_continuous('EstimatedSalary')


# Balance 
plot_continuous('Balance')

# ## Bivariate and Multivariate Analysis

# Here our Target Variable is Churn
df_new['Churn'].value_counts()



# Here 0 is False and 1 is True. Meaning Almost 20% of our clients have exited while Almost 80% are still our customer.
# Let's check the variables that have impacted the result.

df_new


df_new.corr() # Correlation


# Let's plot the correlation table
plt.figure(figsize=(15,5))
sns.heatmap(df_new.corr(),annot=True ,cmap="YlGn" )
plt.show()

df_new.drop('Complain', axis =1, inplace=True)


plt.figure(figsize=(15,5))
sns.heatmap(df_new.corr(),annot=True ,cmap="YlGn" )
plt.show()


# Lets plot and see how various variables ipacted the exit of the customers
# Relationship of  with Exited

# 1. Geography

# 2. Gender

# 3. Tenure

# 4. NumOfProducts

# 5. HasCrCard

# 6. IsActiveMember

# 7. Churn

# 8. Complain ##

# 9. Satisfaction Score

# 10. Card Type



def multivariate_comparision(a, b):
#     a = input("Enter the variable to caopare with the target variable: ")
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.title(f'{a} w.r.t. {b}')
    sns.countplot(x = a, hue = b, palette = ['Red','Green'], data = df_new)
    plt.gca().set_facecolor('lightblue')
    

# compare_with_target('Geography')



multivariate_comparision("Geography", "Churn")
multivariate_comparision('Gender', 'Churn')
# Gender and Satisfaction Score
multivariate_comparision("Gender", "Satisfaction Score")
multivariate_comparision("Tenure", "Churn")
multivariate_comparision("NumOfProducts", "Churn")
multivariate_comparision("Card Type", "Churn")
multivariate_comparision("HasCrCard", "Churn")
multivariate_comparision("Satisfaction Score", "Churn")
multivariate_comparision('IsActiveMember','Churn')


sns.pairplot(data=df_new , corner=True)
plt.show()
# It is difficult to understand anything from this pairplot
# Groupby
df_new.groupby(df_new.Age).Balance.mean()

## Multivariate Analysis

plt.figure(figsize=(15,7))
sns.lineplot(x = df_new["Age"], y = df_new["Balance"], hue=df_new["Churn"], palette = ['Red','Green'],  ci=0).set(title= 'Exit ratio w.r.t Balance and Age')
plt.legend(loc = 'upper right')
plt.gca().set_facecolor('lightblue')
plt.show()


# Similarly lets see the relationship between Gender Geography and Exited
plt.figure(figsize=(20,9))
sns.lineplot(x = df_new["Point Earned"], y = df_new["EstimatedSalary"], hue=df_new["Churn"], palette = ['Red','Green'],  ci=0).set(title= 'Exit ratio w.r.t Balance and Age')
plt.legend(loc = 'upper right')
plt.gca().set_facecolor('lightblue')
plt.show()



# KDE plot
ax = sns.kdeplot(df_new.Balance[(df_new["Churn"] == 0) ],
                color="Red", fill = True)
ax = sns.kdeplot(df_new.Balance[(df_new["Churn"] == 1) ],
                ax = ax, color="Green", fill= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Estimated Salary')
ax.set_title('Distribution of Estimated Salary by churn')


# ## Data Preprocessing
# 
# ### 1. Dimensionality Reduction

# Here we have Three ccategorical columns that we need to transform into continuous variable

transformed = pd.get_dummies(df_new[["Geography", "Gender", "Card Type"]], drop_first = True) 

df_new = pd.concat([df_new, transformed], axis=1)
df_new.drop(['Geography', 'Gender', 'Card Type'],axis = 1, inplace = True)



# ## Feature Scaling
# ### We need to scale down the remaining variables
# The most suitabe sclaer for continuous data would be min-max scaler.


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler to necessary variables
var_to_transform = ['CreditScore','Age','Tenure','Balance','NumOfProducts',
                      'EstimatedSalary','Satisfaction Score', 'Point Earned']
df_new[var_to_transform] = scaler.fit_transform(df_new[var_to_transform])

# ### Splitting the data into train and test
# Here we will train our model in the data and test it against the Churn column. 
X = df_new.drop('Churn', axis=1)
X = X.values
y = df_new["Churn"]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# ## Fitting the data into a model
# ### We will be fitting the data into various models
# ### Below are the Algorithms we will fit out Data to:
# #### 1. Naive Bayes
# #### 2. Decision Tree
# #### 3. Random Forest
# #### 4. Logistic Regression
# #### 5. Adaboost
# #### 6. KNN

# ### 1. Naive Bayes
model = GaussianNB()
model.fit(X_train,y_train)
accuracy_NB = model.score(X_test, y_test)
print("Accuracy of the NB model is: ", accuracy_NB)

pred = model.predict(X_test)
report = classification_report(y_test, pred)
print(report)

plt.figure(figsize = (9,5))
sns.heatmap(confusion_matrix(y_test, pred),annot = True, fmt = "d",linecolor="k",linewidths=3)
plt.title("Gaussian NB Confusion Matrix")
plt.show()


# ##### GaussianNB gave us a accuracy of 83.40% which is decent but still needs imporvements. Let's See how other Algorithms Perform
# 
# ### 2. Decision Trees
# Decision Trees
parameters = {'max_depth': [3, 4, 5, 6, 7, 9, 11],
              'min_samples_split': [2, 3, 4, 5, 6, 7],
              'criterion': ['entropy', 'gini']
             }

model = DecisionTreeClassifier()
gridDecisionTree = RandomizedSearchCV(model, parameters, cv = 3, n_jobs = -1)
gridDecisionTree.fit(X_train, y_train)

print('Mín Split: ', gridDecisionTree.best_estimator_.min_samples_split)
print('Max depth: ', gridDecisionTree.best_estimator_.max_depth)
print('Algorithm: ', gridDecisionTree.best_estimator_.criterion)
print('Score: ', gridDecisionTree.best_score_)

# Using these criterias in our Decision tree, we will predict the result
model_dt = DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 5, max_depth= 7, random_state=0)
model_dt.fit(X_train, y_train)
accuracy_DT = model_dt.score(X_test, y_test)
print("Accuracy of the DT model is: ", accuracy_DT)


## Decison Tree provided an accuracy of 85.8% which is an increment from the GNB model.
def performance_report(model):
    pred = model.predict(X_test)
    report = classification_report(y_test, pred)
    print(report)
performance_report(model_dt)

def plot_cf(a,b):
    model_name = input("Enter Name of the current Model: ")
    plt.figure(figsize = (9,5))
    sns.heatmap(confusion_matrix(a,b),annot = True, fmt = "d",linecolor="k",linewidths=3)
    plt.title(f" {model_name} Confusion Matrix")
    return plt.show()
plot_cf(y_test, pred)


# ##### We can Clearly see a slight improvement in the TP in the Confusion Matrix. TN also increased slightly

# ### 3. Random Forest

model_RF = RandomForestClassifier(n_estimators=100 , oob_score = True, n_jobs = -1,
                                  random_state =42,
                                  max_leaf_nodes = 25)

model_RF.fit(X_train, y_train)
# cv_scores = cross_val_score(model_RF, X_train, y_train, cv= 10) # Five fold cross validation
# print("Cross-Validation Scores:", cv_scores)
# average_cv_score = np.mean(cv_scores)
# print("Average Cross-Validation Score:", average_cv_score)
accuracy_RF = model_dt.score(X_test, y_test)
print("Accuracy of the RFC model is: ", accuracy_RF)


# Even with 10 folds cross validatin the result was same
# ![image.png](attachment:image.png)

#Plotting Performance Report
performance_report(model_RF)


# Plotting Heatmap or Confusion Matric
plot_cf(y_test, pred)

# The Accuracy produced by Both Decision Trees and Random Forest is Exactly same. Lets plot the Roc Curve     
y_pred_prob = model_RF.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr, tpr, label='Random Forest Classifier',color = "r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve',fontsize=16)
plt.show()


# ### 4. Logistic Regression


model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)
accuracy_LR = model_LR.score(X_test,y_test)
print("Logistic Regression accuracy is :",accuracy_LR)

performance_report(model_LR)
plot_cf(y_test, pred)


# ##### Here, Despite decrease in accuracy, we can see a slight improvement in the prediction of True Positives 
# While the prediction False Positive Decreased. Prediction of True Negative went down in Logistic Regression.

y_pred_prob = model_LR.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr, tpr, label='Logistic Regression Classifier',color = "r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve',fontsize=16)
plt.show()
