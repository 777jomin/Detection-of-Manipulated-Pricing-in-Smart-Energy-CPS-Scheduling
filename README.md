# Detection-of-Manipulated-Pricing-in-Smart-Energy-CPS-Scheduling
According to the problem statement of the coursework, we are provided with the data of 5 users who own 10 smart home appliances each. This data indicates that each user is given 10 tasks to complete with the smart appliances for which proper scheduling is necessary. We are also provided with 10000 predictive guideline price curves as training data that gives the pricing details for each hour of the day and the corresponding label indicating whether the pricing curve is normal or abnormal. By using this data, a suitable model needs to be designed for classifying the data to be scheduled into normal or abnormal, which is done to effectively predict a pricing attack. A set of 100 pricing curves is provided to be used as testing data for which labelling (normal or abnormal) must be determined using the model designed previously. For the price curves that are classified to be abnormal, we must minimize the cost by developing a Linear Programming energy solution based on the abnormal predictive guideline price curve and plot the obtained scheduling results displaying the hourly energy usage of the 5 users. The minimized schedule obtained as a result of our solution will be the corresponding normal scheduling for each abnormal price curve. 




TO RUN

Clone the git repository contents to a single folder.

This can be done by using the command "git clone https://github.com/777jomin/Detection-of-Manipulated-Pricing-in-Smart-Energy-CPS-Scheduling".

The contents of the repository Detection-of-Manipulated-Pricing-in-Smart-Energy-CPS-Scheduling will now be available on your local machine.

Execute the 'run.sh' file to perform complete execution of code and obtain the bar chart results in '/Plots',the Testing results in TestingResults.txt and prediction results in PredictionsOnly.txt.





PS: If you get an error saying "./run.sh: line 2: $'\r': command not found" or "command not found" or ": not found"

   please run the following command before executing 'run.sh'
   
   sed -i -e 's/\r$//' run.sh
