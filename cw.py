import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from pulp import *
import matplotlib.pyplot as plt


#Reading the provided training data

trainingDF = pd.read_csv('TrainingData.txt', header=None)
y = trainingDF[24].tolist()
trainingDF = trainingDF.drop(24, axis=1)
x = trainingDF.values.tolist()

#Storing the training data before splitting

x = np.array(x)
y = np.array(y)
x_train_full = x
y_train_full = y

#Reading the testing data to make predictions

testingDF = pd.read_csv('TestingData.txt', header=None)
x_classify = testingDF.values.tolist()

#Splitting of training data to test the algorithm

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(len(x_train),len(x_test),len(y_train),len(y_test))

#Normalising the value between 0 and 1

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_classify = scaler.transform(x_classify)
x_train_full = scaler.transform(x_train_full)

#Performing Linear Discriminant Analysis

lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
y_pred = lda.predict(x_classify)
y_pred = [int(x) for x in y_pred]
print("\nAccuracy for LDA classifier on full Training Dataset:",lda.score(x_train_full, y_train_full))

#Displaying results of testing and training

print("Testing accuracy for 25% of training data:",lda.score(x_test, y_test))
print("Training accuracy:",lda.score(x_train, y_train))

#Classifying the testing data and getting labels

yclas_pred = lda.predict(x_classify)
yclas_pred = [int(i) for i in yclas_pred]
predDF = pd.DataFrame({'Prediction': y_pred})
testingDF = testingDF.join(predDF)
testingDF.to_csv("TestingResults.txt", header=None, index=None)
predDF.to_csv("PredictionsOnly.txt", header=None, index=None)
print("\nPredictions saved in output file TestingResults.txt")
for j in range(0,100):
  print("DAY => "+str(j+1)+"\tPrediction => "+str(yclas_pred[j]) )

def Readingdata():

    #Reading the user task information from given excel file
    
    excelFile = pd.read_excel ('COMP3217CW2Input.xlsx', sheet_name = 'User & Task ID')
    taskName = excelFile['User & Task ID'].tolist()
    readyTime = excelFile['Ready Time'].tolist()
    deadline = excelFile['Deadline'].tolist()
    maxEnergyPerHour = excelFile['Maximum scheduled energy per hour'].tolist()
    energyDemand = excelFile['Energy Demand'].tolist()
    tasks = []
    taskNames = []

    for k in range (len(readyTime)):
        task = []
        task.append(readyTime[k])
        task.append(deadline[k])
        task.append(maxEnergyPerHour[k])
        task.append(energyDemand[k])
        taskNames.append(taskName[k])
    
        tasks.append(task)
     
    #Reading the output of testing data
    
    testingDF = pd.read_csv('TestingResults.txt', header=None)
    y_labels = testingDF[24].tolist()
    testingDF = testingDF.drop(24, axis=1)
    x_data = testingDF.values.tolist()
    
    return tasks, taskNames, x_data, y_labels


#The hourly energy use for the community is plotted

def plot(model, count, cost):
    hours = [str(x) for x in range(0, 24)]
    pos = np.arange(len(hours))
    users = ['user1', 'user2', 'user3', 'user4', 'user5']
    color_list = ['midnightblue','mediumvioletred','mediumturquoise','gold','linen']
    plot_list = []
    to_plot = []
    
    #Creating lists to plot the usage
    
    for user in users:
        temp_list = []
        for hour in hours:
            hour_list_temp = []
            task_count = 0
            for var in model.variables():
                if user == var.name.split('_')[0] and str(hour) == var.name.split('_')[2]:
                    task_count += 1                   
                    hour_list_temp.append(var.value())
            temp_list.append(sum(hour_list_temp))
        plot_list.append(temp_list)

    #Displaying barchart stacked by the user.
    
    plt.bar(pos,plot_list[0],color=color_list[0],edgecolor='black',bottom=0)
    plt.bar(pos,plot_list[1],color=color_list[1],edgecolor='black',bottom=np.array(plot_list[0]))
    plt.bar(pos,plot_list[2],color=color_list[2],edgecolor='black',bottom=np.array(plot_list[0])+np.array(plot_list[1]))
    plt.bar(pos,plot_list[3],color=color_list[3],edgecolor='black',bottom=np.array(plot_list[0])+np.array(plot_list[1])+np.array(plot_list[2]))
    plt.bar(pos,plot_list[4],color=color_list[4],edgecolor='black',bottom=np.array(plot_list[0])+np.array(plot_list[1])+np.array(plot_list[2])+np.array(plot_list[3]))
    
    plt.xticks(pos, hours)
    plt.xlabel('Hour')
    plt.ylabel('Energy Usage (kW)')
    plt.title('Energy Usage Per Hour For All Users\nDay %i\nMinimized Cost = %.15f'%(count ,cost))
    plt.legend(users,loc=0)
    figure= plt.gcf()
    figure.savefig('Plots/'+str(count)+'.png',bbox_inches='tight')
    plt.clf()

    return plot_list
    
#function for computing the linear problem and solving it
def creatingLPModel(tasks, taskNames):
 
    vars = []
    c = []
    eq = []
    
    #creating the LP problem model for Minimization    
    model = LpProblem(name="scheduling-problem", sense=LpMinimize)
    
    #Looping through list of tasks
    for ind, task in enumerate(tasks):
        n = task[1] - task[0] + 1
        temp = []
        #Looping between ready_time and deadline for each task
        #Creation of LP variables with given constraints and unique names
        for i in range(task[0], task[1] + 1):
            x = LpVariable(name=taskNames[ind]+'_'+str(i), lowBound=0, upBound=task[2])
            temp.append(x)
        vars.append(temp)
        
    #Creating the objective function to minimize the price and adding to the model
    for i, task in enumerate(tasks):
        for var in vars[i]:
            price = price_list[int(var.name.split('_')[2])]
            c.append(price * var)
    model += lpSum(c)
              
    #Introducing additional constraints to the model      
    for i, task in enumerate(tasks):
        temp = []
        for var in vars[i]:
            temp.append(var)
        eq.append(temp)
        model += lpSum(temp) == task[3]
    
    #Return model to the main function
    return model
    
tasks, task_names, x_data, y_labels = Readingdata()
answerlist=[]

for index, price_list in enumerate(x_data):
  #Scheduling and plotting the abnormal guideline pricing curves  
  if y_labels[index] == 1:
  #Solving the returned LP model from the scheduling solution
        model = creatingLPModel(tasks, task_names)
        answer = model.solve()
        answerlist.append(answer)
        #Print LP model stats
        #Ploting the hourly usage for the scheduling solution
        plot(model,index+1,value(model.objective))
        print(LpStatus[answer], value(model.objective))
        
print("Total number of abnormal charts =", len(answerlist))
print("The plotted bar charts for the scheduling solution are available in the folder '/Plots'")
        