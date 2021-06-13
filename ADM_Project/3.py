
import pandas as pd
import numpy as np
import scipy.stats as spstats
import random
from scipy import interpolate

# for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
  
import pydotplus
import graphviz
import pydotplus
#%matplotlib inline

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from deap import creator, base, tools, algorithms
from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image 

from sklearn.datasets import make_classification
from sklearn import linear_model
from feature_selection_ga import FeatureSelectionGA, FitnessFunction

import sys
from scoop import futures

#defining various steps required for the genetic algorithm
def initilization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat,dtype=bool)
        chromosome[:int(0.3*n_feat)]=False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(population):
    scores = []
    for chromosome in population:
        logmodel.fit(X_train.iloc[:,chromosome],y_train)
        predictions = logmodel.predict(X_test.iloc[:,chromosome])
        scores.append(accuracy_score(y_test,predictions))
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds,:][::-1])

def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen

def crossover(pop_after_sel):
    population_nextgen=pop_after_sel
    for i in range(len(pop_after_sel)):
        child=pop_after_sel[i]
        child[3:7]=pop_after_sel[(i+1)%len(pop_after_sel)][3:7]
        population_nextgen.append(child)
    return population_nextgen

def mutation(pop_after_cross,mutation_rate):
    population_nextgen = []
    for i in range(0,len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[j]= not chromosome[j]
        population_nextgen.append(chromosome)
    #print(population_nextgen)
    return population_nextgen

def generations(size,n_feat,n_parents,mutation_rate,n_gen,X_train,
                                   X_test, y_train, y_test):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        print(scores[:2])
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross,mutation_rate)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo,best_score

if __name__ == '__main__':

    #reading the dataset
    data = pd.read_csv('./online_shoppers_intention.csv')

    # checking the shape of the data
    print(data.shape)


    # checking the head of the data : first 5 columns
    print(data.head())

    #for decision Tree# describing the data, displays the stats of all the attributes.
    print(data.describe())

    #ANALYZING THE ATTRIBUTES
    
    plt.rcParams['figure.figsize'] = (18, 7)
    
    # checking the Distribution of customers on Revenue
    plt.subplot(1, 2, 1)
    sns.countplot(data['Weekend'], palette = 'pastel')
    plt.title('Buy or Not', fontsize = 30)
    plt.xlabel('Revenue or not', fontsize = 15)
    plt.ylabel('count', fontsize = 15)


    # checking the Distribution of customers on Weekend
    plt.subplot(1, 2, 2)
    sns.countplot(data['Weekend'], palette = 'inferno')
    plt.title('Purchase on Weekends', fontsize = 30)
    plt.xlabel('Weekend or not', fontsize = 15)
    plt.ylabel('count', fontsize = 15)

    plt.show()

    # pie chart for analyzing the relation between special days and Month

    size = [3364, 2998, 1907, 1727, 549, 448, 433, 432, 288, 184]
    colors = ['cyan', 'pink', 'crimson', 'orange','lightgreen', 'lightblue','yellow', 'lightgreen', 'violet', 'magenta']
    labels = "May", "November", "March", "December", "October", "September", "August", "July", "June", "February"
    explode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    circle = plt.Circle((0, 0), 0.6, color = 'white')

    #plt.subplot(1, 2, 2)
    plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
    plt.title('Special Days', fontsize = 30)
    p = plt.gcf()
    p.gca().add_artist(circle)
    plt.axis('off')
    plt.legend()

    plt.show()

    # product related duration vs revenue

    plt.rcParams['figure.figsize'] = (18, 15)

    plt.subplot(2, 2, 1)
    sns.boxenplot(data['Revenue'], data['Informational_Duration'], palette = 'rainbow')
    plt.title('Info. duration vs Revenue', fontsize = 15)
    plt.xlabel('Info. duration', fontsize = 15)
    plt.ylabel('Revenue', fontsize = 15)

    # product related duration vs revenue

    plt.subplot(2, 2, 2)
    sns.boxenplot(data['Revenue'], data['Administrative_Duration'], palette = 'pastel')
    plt.title('Admn. duration vs Revenue', fontsize = 15)
    plt.xlabel('Admn. duration', fontsize = 15)
    plt.ylabel('Revenue', fontsize = 15)

    # product related duration vs revenue

    plt.subplot(2, 2, 3)
    sns.boxenplot(data['Revenue'], data['ProductRelated_Duration'], palette = 'dark')
    plt.title('Product Related duration vs Revenue', fontsize = 15)
    plt.xlabel('Product Related duration', fontsize = 15)
    plt.ylabel('Revenue', fontsize = 15)

    # exit rate vs revenue

    plt.subplot(2, 2, 4)
    sns.boxenplot(data['Revenue'], data['ExitRates'], palette = 'spring')
    plt.title('ExitRates vs Revenue', fontsize = 15)
    plt.xlabel('ExitRates', fontsize = 15)
    plt.ylabel('Revenue', fontsize = 15)

    plt.subplot_tool()
    plt.show()

    

    #DISCRETIZATION

    #Perform bucketing using the pd.cut() function on the marks column and display the top 10 columns.
    #The cut() function takes parameters such as x, bins, and labels. 
    #Here, we have used only three parameters. Add the following code to implement this:
    # df['bucket']=pd.cut(df['marks'],5,labels=['Poor','Below_average','Average','Above_Average','Excellent'])'''

    #TO check : df.head(10)

    #1.Administrative

    #data['Administrative_bucket']=pd.cut(data['Administrative'],6,labels=[5,10,15,20,25,30])
    data['Administrative_buckets'] = pd.cut(data['Administrative'], bins=[0,10,15,20,25,30],labels =['VeryLow','Low','Medium','High','VeryHigh'],include_lowest=True)
    print(data)
      
    # We can check the frequency of each bin
    print(data['Administrative_buckets'].unique())

    #-----------------------------------------------------------------------------------------------------------------------------------
    #2.Administrative Duration

    data['AdministrativeDuration_buckets'] = pd.cut(data['Administrative_Duration'], bins=[0,500,1000,1500,2000,2500],labels =['VeryLow','Low','Medium','High','VeryHigh'],include_lowest=True)
    print(data)
      
    # We can check the frequency of each bin
    print(data['AdministrativeDuration_buckets'].unique())

    #-----------------------------------------------------------------------------------------------------------------------------------

    #3.Informational

    data['Informational_buckets'] = pd.cut(data['Informational'], bins=[0,10,15,20,25],labels=['VeryLow','Low','Medium','High'],include_lowest=True)
    print(data)
      
    # We can check the frequency of each bin
    print(data['Informational_buckets'].unique())

    #-----------------------------------------------------------------------------------------------------------------------------------
    #4.Informational Duration

    data['InformationalDuration_buckets'] = pd.cut(data['Informational_Duration'], bins=[-1,1000,2000,3000],labels =['Low','Medium','High'],include_lowest=True)
    print(data)
      
    # We can check the frequency of each bin
    print(data['InformationalDuration_buckets'].unique())

    #-----------------------------------------------------------------------------------------------------------------------------------
    #5.Product Related

    data['ProductRelated_buckets'] = pd.cut(data['ProductRelated'], bins=[0,200,400,600,800],labels=['VeryLow','Low','Medium','High'],include_lowest=True)
    print(data)
      
    # We can check the frequency of each bin
    print(data['ProductRelated_buckets'].unique())

    #-----------------------------------------------------------------------------------------------------------------------------------
    #6.Product Related Duration

    data['ProductRelatedDuration_buckets'] = pd.cut(data['ProductRelated_Duration'], bins=[0,15000,30000,45000,60000,75000],labels =['VeryLow','Low','Medium','High','VeryHigh'],include_lowest=True)
    print(data)
      
    # We can check the frequency of each bin
    print(data['ProductRelatedDuration_buckets'].unique())

    #-----------------------------------------------------------------------------------------------------------------------------------
    #7.BounceRate

    data['BounceRates_buckets'] = pd.cut(data['BounceRates'], bins=[0,0.015,0.025,0.05,0.1,0.2],labels =['VeryLow','Low','Medium','High','VeryHigh'],include_lowest=True)
    print(data)
      
    # We can check the frequency of each bin
    print(data['BounceRates_buckets'].unique())

    #-----------------------------------------------------------------------------------------------------------------------------------
    #8.ExitRate

    data['ExitRates_buckets'] = pd.cut(data['ExitRates'], bins=[0,0.015,0.025,0.05,0.1,0.2],labels =['VeryLow','Low','Medium','High','VeryHigh'],include_lowest=True)
    print(data)
      
    # We can check the frequency of each bin
    print(data['ExitRates_buckets'].unique())

    #-----------------------------------------------------------------------------------------------------------------------------------
    #9.PageValue

    data['PageValues_buckets'] = pd.cut(data['PageValues'], bins=[0,100,200,300,400],labels=['VeryLow','Low','Medium','High'],include_lowest=True)
    print(data)
      
    # We can check the frequency of each bin
    print(data['PageValues_buckets'].unique())

    #-----------------------------------------------------------------------------------------------------------------------------------
    #10.SpecialDay

    data['SpecialDay_buckets'] = pd.cut(data['SpecialDay'], bins=[0,0.15,0.35,0.5,0.75,1.0],labels =['VeryLow','Low','Medium','High','VeryHigh'],include_lowest=True)
    print(data)
      
    # We can check the frequency of each bin
    print(data['SpecialDay_buckets'].unique())

    #-----------------------------------------------------------------------------------------------------------------------------------

    #CREATING A NEW DATASET WITH DISCRETIZED ATTRIBUTES BY REMOVING THE ADDITIONAL NUMERICAL ATTRIBUTES.
    
    print(data.shape)

    #Dropping the numeric attributes after adding their corresponding discretized counter-attributes
    data.drop(['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration',
'BounceRates','ExitRates', 'PageValues','SpecialDay'],inplace=True, axis=1)

    data.to_csv("./Discretized_online_shoppers_intention.csv",index=False)

    #END OF DISCRETIZATION.'''

    df = pd.read_csv('./Discretized_online_shoppers_intention.csv', sep=',')

    #Label Encoding
    labelencoder = LabelEncoder()

    df["Administrative_Val"]=labelencoder.fit_transform(df["Administrative_buckets"])
    
    df["Administrative_Duration_Val"]=labelencoder.fit_transform(df["AdministrativeDuration_buckets"])

    df["Informational_Val"]=labelencoder.fit_transform(df["Informational_buckets"])

    df["Informational_Duration_Val"]=labelencoder.fit_transform(df["InformationalDuration_buckets"])

    df["ProductRelated_Val"]=labelencoder.fit_transform(df["ProductRelated_buckets"])

    df["ProductRelated_Duration_Val"]=labelencoder.fit_transform(df["ProductRelatedDuration_buckets"])

    df["BounceRates_Val"]=labelencoder.fit_transform(df["BounceRates_buckets"])

    df["ExitRates_Val"]=labelencoder.fit_transform(df["ExitRates_buckets"])

    df["PageValues_Val"]=labelencoder.fit_transform(df["PageValues_buckets"])

    df["Month_Val"]=labelencoder.fit_transform(df["Month"])

    df["VisitorType_Val"]=labelencoder.fit_transform(df["VisitorType"])

    df["SpecialDay_Val"]=labelencoder.fit_transform(df["SpecialDay_buckets"])

    df["Weekend_Val"]=labelencoder.fit_transform(df["Weekend"])

    df["Revenue_Val"]=labelencoder.fit_transform(df["Revenue"])

    #Dropping the non-encoded attributes after adding their corresponding label encoded counter-attributes
    data.drop(['Administrative_buckets','AdministrativeDuration_buckets','Informational_buckets',
'InformationalDuration_buckets','ProductRelated_buckets','ProductRelatedDuration_buckets',
'BounceRates_buckets','ExitRates_buckets', 'PageValues_buckets','Month',
'VisitorType','SpecialDay_buckets','Weekend','Revenue'],inplace=True, axis=1)

    print(data.head())

   
    #FEATURE SELECTION USING GA
    # Read in data from CSV

    #contains all the features of the latest dataframe
    feature_cols = ['Administrative_Val','Administrative_Duration_Val','Informational_Val','Informational_Duration_Val','ProductRelated_Val','ProductRelated_Duration_Val','BounceRates_Val','ExitRates_Val','PageValues_Val','SpecialDay_Val','OperatingSystems','Browser','Region','TrafficType','VisitorType_Val','Weekend_Val','Month_Val']   
    X = df[feature_cols] # Features
    y = df.Revenue_Val  #Target Feature

    #print(X)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

    #GA
    X, y = make_classification(n_samples=12238, n_features=18, n_classes=2, n_informative=6, n_redundant=0, n_repeated=0,random_state=1)

    model = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    fsga = FeatureSelectionGA(model,X,y, ff_obj = FitnessFunction())
    pop = fsga.generate(20)
    
    #print(pop)
    
    #selecting 5 features from the FeatureSelection set.
    features = [0]*18;
    col=0;
    for row in range(0,20) :
        if(pop[row][col]!=1 and col==17):
            break;
        features[col]=1;
        if(col==17):
            col=0

    print(features)

    
    #DECISION TREE GENERATION

    X = df[feature_cols]    #Features
    y = df.Revenue_Val      #Target Feature

    #Splitting the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    f = open("Treefile.txt","x")
    text_representation = tree.export_text(clf, feature_names=feature_cols)
    print(text_representation)
    f.write(text_representation)
    f.close()

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,filled=True, rounded=True,special_characters=True,feature_names = feature_cols,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('DecisionTree.png')
    Image(graph.create_png())

   
    





