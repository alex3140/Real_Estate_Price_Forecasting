import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import AgglomerativeClustering
#from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('C:\\Users\\alex314\\Desktop\\Allegheny\\alleghenycountymasterfile05272016.csv', 
                    low_memory = False, dtype = {'PROPERTYZIP': 'str', 
                                             'PROPERTYHOUSENUM' : 'str'})
                          
data1 = data[(data["SALEDESC"].isin(["VALID SALE", "OTHER VALID"])) 
            & (data["USEDESC"].isin(["SINGLE FAMILY", "TOWNHOUSE", "ROWHOUSE"]))
            & (data["CONDITION"].isin([1,2,3,4,7]))
            & (data["CDUDESC"]).isin(["EXCELLENT", "VERY GOOD", "GOOD", 
                                      "AVERAGE", "FAIR"])]
                                             
#data1["SINGLEFAMILY"]=pd.get_dummies(data1.USEDESC)["SINGLE FAMILY"]
#data1["TOWNHOUSE"]=pd.get_dummies(data1.USEDESC)["TOWNHOUSE"]
data1["SALEDATE"] = pd.to_datetime(data1.SALEDATE, format = "%m-%d-%Y")
data1["SALEYEAR"] = pd.Series([data1.SALEDATE[idx].year for idx in data1.index],
                                index = data1.index)
data1["SALEMONTH"] = pd.Series([data1.SALEDATE[idx].month for idx in data1.index],
                                index = data1.index)
data2 = data1[data1["SALEYEAR"].isin(range(2008,2017))]
obj = data2.groupby(["SALEYEAR","SALEMONTH"]).SALEPRICE

fig = plt.figure()
dates = pd.date_range("1/1/2008", periods = obj.mean().shape[0], freq="MS")
median, = plt.plot_date(dates, obj.median(), 'b-', tz = None, 
                    xdate = True, ydate = False, label = 'Median Price')
mean, = plt.plot_date(dates, obj.mean(), 'r-', label = 'Mean Price')
plt.legend(handles=[median, mean])
plt.title("Mean and Median Sale Price per Month, $")

data2 = data2.drop([384712])#record with missing house number
data2["ADDRESS"] = pd.Series(
                    [data2.PROPERTYHOUSENUM[idx] + " "
                    + str(data2.PROPERTYADDRESS[idx]) + " PA " 
                    + data2.PROPERTYZIP[idx]
                    for idx in data2.index], index = data2.index)

data2 = data2[data2.SALEPRICE < 500000]
fig = plt.figure()
sns.boxplot(x = data2.ROOF, y = data2.SALEPRICE)#~50 missing vals
fig = plt.figure()
sns.boxplot(x = data2.BASEMENT, y = data2.SALEPRICE)#~1600 missing vals
fig = plt.figure()
sns.boxplot(x = data2.BSMTGARAGE, y = data2.SALEPRICE)#~1600 missing vals
fig = plt.figure()
sns.boxplot(x = data2.CONDITION, y = data2.SALEPRICE)
fig = plt.figure()
sns.boxplot(x = data2.CDUDESC, y = data2.SALEPRICE)
fig = plt.figure()
sns.boxplot(x = data2.HALFBATHS, y = data2.SALEPRICE) #~300 missing vals
fig = plt.figure()
sns.boxplot(x = data2.FIREPLACES, y = data2.SALEPRICE)#~4000 missing vals
fig = plt.figure()
sns.boxplot(x = data2.EXTERIORFINISH, y = data2.SALEPRICE)#~4000 missing vals

#Include TAX?                    
data3 = data2[['PARID', 
                'ADDRESS', 
                'PROPERTYZIP', 
                'SCHOOLCODE', 
                'LOTAREA', 
                'SALEDATE', 
                'SALEYEAR', 
                'SALEMONTH',
                'SALEPRICE', 
                'PREVSALEDATE', 
                'PREVSALEPRICE', 
                'FAIRMARKETTOTAL', 
                'YEARBLT', 
                'ROOF', 
                'EXTERIORFINISH',
                'GRADE', 
                'CONDITION',
                'BEDROOMS', 
                'FULLBATHS', 
                'HALFBATHS',  
                'FIREPLACES', 
                'BSMTGARAGE', 
                'FINISHEDLIVINGAREA', 
                'STYLEDESC']]
                    
data3 = data3[data3.GRADE.notnull()]
geodata = pd.read_csv(
 'C:\\Users\\alex314\\Desktop\\Allegheny\\April2016CentroidBlock.csv', 
                    low_memory = False)
#In this data the labels for Lat and Long are flipped
mygeodata = geodata[["PIN", "Latitude", "Longitude", "geo_id_tra"]]
mygeodata = mygeodata.rename(columns = {"PIN":"PARID", "Latitude" : "LONGITUDE", 
                             "Longitude" : "LATITUDE", "geo_id_tra" : "GEOID"})
data4 = pd.merge(data3, mygeodata, how = 'left', on = 'PARID')

#Log variables
#modeldata["LOGSALEPRICE"]=pd.Series([np.log(modeldata.SALEPRICE[idx]) 
#                            for idx in modeldata.index], index=modeldata.index)

data4["AGE"] = data4.SALEYEAR - data4.YEARBLT
#data4.CONDITION[data4.CONDITION == 4.0] = 5.0
#data4.CONDITION[data4.CONDITION == 3.0] = 4.0
#data4.CONDITION[data4.CONDITION == 2.0] = 3.0
data4.CONDITION[data4.CONDITION == 7.0] = 1.0
fig = plt.figure()
sns.boxplot(x = data4.CONDITION, y = data4.SALEPRICE)
 
data4["GRADENUM"] = 0
data4["GRADENUM"][data4.GRADE.isin(["A", "A-", "A+"])] = 1
data4["GRADENUM"][data4.GRADE == "B+"] = 1.7
data4["GRADENUM"][data4.GRADE == "B"] = 2.0
data4["GRADENUM"][data4.GRADE == "B-"] = 2.3
data4["GRADENUM"][data4.GRADE == "C+"] = 2.7
data4["GRADENUM"][data4.GRADE == "C"] = 3.0
data4["GRADENUM"][data4.GRADE == "C-"] = 3.3
data4["GRADENUM"][data4.GRADE.isin(["D+", "D"])] = 4.0
data4["GRADENUM"][data4.GRADE == "D-"] = 4.3
fig = plt.figure()
sns.boxplot(x = data4.GRADENUM, y = data4.SALEPRICE)

tract_data = pd.read_csv(
    'C:\\Users\\alex314\\Desktop\\Allegheny\\tract_data.csv',
    dtype={"GEOID" : "str"})
tract_data = tract_data.drop("Unnamed: 0", axis = 1)
data5 = pd.merge(data4, tract_data, how = "left", on = "GEOID")
                                      
#Add HALFBATHS?
modeldata = data5[(data5.SALEPRICE < 350000)
                    & (data5.SALEPRICE > 10000)
                    & (data5.LOTAREA < 20000) 
                    & (data5.BEDROOMS > 0)
                    & (data5.BEDROOMS < 7)
                    & (data5.FULLBATHS > 0)
                    & (data5.FULLBATHS < 6)
                    & (data5.FINISHEDLIVINGAREA < 3500)
                    & (2 * data5.FAIRMARKETTOTAL > data5.SALEPRICE)
                    & (data5.FAIRMARKETTOTAL < 2 * data5.SALEPRICE)
                    & (data5. AGE >= 0)
                    & (data5.AGE < 130)
                    & (data5.GRADENUM > 0)]
                    #& (data5.MED_FAM_INCOME14 < 160000)]

#Boxplots
fig = plt.figure()
sns.boxplot(y = modeldata.SALEPRICE, orient='h')
fig = plt.figure()
ax = sns.boxplot(x = modeldata["OWNEROCCRATE"])
fig = plt.figure()
ax = sns.boxplot(x = modeldata["MEANTIMECOMMUTE"])
fig = plt.figure()
sns.boxplot(x = modeldata.SALEYEAR, y = modeldata.SALEPRICE)
fig = plt.figure()
sns.boxplot(modeldata.FINISHEDLIVINGAREA)
fig = plt.figure()
sns.boxplot(modeldata.LOTAREA)
fig = plt.figure()
sns.boxplot(x=modeldata.SCHOOLCODE, y=modeldata.SALEPRICE)
fig = plt.figure()
sns.boxplot(x = modeldata.ROOF, y = modeldata.SALEPRICE)
fig = plt.figure()
sns.boxplot(x = modeldata.AGE)
fig = plt.figure()
sns.boxplot(x = data5.MED_FAM_INCOME14)
fig = plt.figure()
sns.boxplot(x = modeldata.POPDENS)
fig = plt.figure()
sns.boxplot(x = modeldata.PERCENTBACHELORS)
fig = plt.figure()
sns.boxplot(x = modeldata.PERCENTBLACK)

g = sns.PairGrid(modeldata, vars = ['SALEPRICE', 'AGE'])
g = g.map_diag(plt.hist, edgecolor = "w")
g = g.map_offdiag(plt.scatter, edgecolor = "w", s = 15)
                  
#Compute stat. significance?!  
np.corrcoef([modeldata.SALEPRICE, modeldata.CONDITION, modeldata.OWNEROCCRATE])
                                                                                                            
#Map
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(modeldata.LONGITUDE, modeldata.LATITUDE, modeldata.SALEPRICE)
zipcode_list = ['15202', '15209', '15212', '15214', '15229', '15116', '15206',
              '15235', '15146', '15221', '15208', '15218', '15217', '15207',
              '15120', '15227', '15226', '15234', '15210', '15216', '15211',
               '15220', '15205', '15204', '15136']               

zipdata = modeldata[modeldata.PROPERTYZIP.isin(zipcode_list)]

alist = []
counts = modeldata.GEOID.value_counts()
for geoid in counts.index:
    if counts[geoid] < 50:
        alist.append(geoid) 
geooutliers = modeldata[modeldata.GEOID.isin(alist)]

fig = plt.figure()
plt.scatter(modeldata.LONGITUDE, modeldata.LATITUDE)
#plt.scatter(zipdata.LONGITUDE, zipdata.LATITUDE, color='r')
#d = modeldata[modeldata.SCHOOLCODE.isin([48,49,50])]
#plt.scatter(d.LONGITUDE, d.LATITUDE, color = 'yellow')
plt.scatter(zipdata.LONGITUDE, zipdata.LATITUDE, color='r')
#plt.scatter(geooutliers.LONGITUDE, geooutliers.LATITUDE, color='yellow')
plt.title("Property locations in Allegheny County")
plt.ylabel("Latitude")
plt.xlabel("Longitude")
                                      
#Cluster Analysis (Hierarchical Clustering)                             
clusterModel = AgglomerativeClustering(affinity='euclidean', linkage='ward',
                                                n_clusters=5)
 
clusterdata = modeldata[["AGE", "FINISHEDLIVINGAREA", 
                            "INCOME", "LATITUDE", "LONGITUDE"]]                                                 
clusterModel.fit(clusterdata)
plt.figure()
plt.scatter(clusterdata.LONGITUDE, clusterdata.LATITUDE, c=clusterModel.labels_,
            cmap = plt.cm.spectral)
            
modeldata["CLUSTER"] = pd.Series(clusterModel.labels_, index = modeldata.index)

#Model Building
features = ['FINISHEDLIVINGAREA', 
            'LOTAREA', 
            'BEDROOMS', 
            'FULLBATHS',
            'SCHOOLCODE',
            'CONDITION',
            'GRADENUM',
            'AGE',
            'SALEMONTH',
            'SALEYEAR',
            'LATITUDE',
            'LONGITUDE',
            'POPDENS',
            'PERCENTPOVERTY',
            'PERCENTBLACK',
            'PERCENTBACHELORS',
            'MED_FAM_INCOME14',
            'OWNEROCCRATE',
            'VACANCYRATE',
            'MEANTIMECOMMUTE',
            'ROOF',
            'EXTERIORFINISH']
fig = plt.figure()
sns.boxplot(x = modeldata.CONDITION, y = modeldata.SALEPRICE)

plt.figure()
plt.scatter(modeldata.VACANCYRATE, modeldata.SALEPRICE)
                      
#Quality of schools based on schoolcode?
testdata = modeldata[['SALEPRICE']+features].dropna()
X, Y = testdata[features], testdata['SALEPRICE']
Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(
        X, Y, test_size=.1, random_state = 777)
        
#Gradient tree boosting model          
boost = GradientBoostingRegressor(learning_rate = .01, subsample = .5,
                                n_estimators = 5000, max_depth = 7,
                                max_features = "sqrt")
boost.fit(Xtrain, Ytrain)
RMSEs_boost = [mean_squared_error(Ytrain, boost.predict(Xtrain))**.5,
               mean_squared_error(Ytest, boost.predict(Xtest))**.5]
MedAPE_test = (abs(Ytest - boost.predict(Xtest)) / Ytest).median()
MeanAPE_test = (abs(Ytest - boost.predict(Xtest)) / Ytest).mean() 

percent = sum(abs(Ytest - boost.predict(Xtest)) / Ytest < .1) / float(Ytest.shape[0])

print "***(Stochastic) Gradient Boosting Tree Model***"
print "Median Absolute Percentage Error: %s" %(round(MedAPE_test,4) * 100), "%"
print "Mean Absolute Percentage Error:", round(MeanAPE_test,5) * 100, "%"
print "Percentage of predictions with error below 10%:", round(percent, 4) * 100, "%"
print "Root MSE on train set: %s, Root MSE on test set: %s" \
        %(round(RMSEs_boost[0], 1),round(RMSEs_boost[1], 1))
print "R-square is %s" %(round(r2_score(Ytest,boost.predict(Xtest)), 4) * 100),"%"

featImportances = boost.feature_importances_
pos = np.arange(len(features))
pairs = zip(features, featImportances)
sorted_pairs = sorted(pairs, key = lambda pair: pair[1])
features_sorted, featImportances_sorted = zip(*sorted_pairs)
fig, ax = plt.subplots()
plt.barh(pos, featImportances_sorted, 1, color = "blue")
plt.yticks(pos,features_sorted)
ax.set_title('Gradient Boosting: Relative Feature Importance')

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.scatter(np.array(Ytest), boost.predict(Xtest))
ax.plot(np.arange(testdata.SALEPRICE.min(),testdata.SALEPRICE.max(), .1), 
 np.arange(testdata.SALEPRICE.min(),testdata.SALEPRICE.max(),.1), color='navy')
ax.set_title('Predicted price vs. Actual price')
ax.set_ylabel('Predicted price, $')
ax2 = fig.add_subplot(2, 1, 2)
ax2.scatter(np.array(Ytest), np.array(Ytest)-boost.predict(Xtest))
ax2.set_title('Residuals vs. actual price')
ax2.set_xlabel('Actual price, $')
ax2.set_ylabel('Residual, $')

#Study large residuals
Ytest[abs(np.array(Ytest)-boost.predict(Xtest))>100000]

#Study various factors
fig = plt.figure()
plt.scatter(Xtest.AGE, np.array(Ytest)-boost.predict(Xtest))
ax2.set_title('Residuals vs. age')

fig = plt.figure()
plt.scatter(Xtest.PERCENTPOVERTY, np.array(Ytest)-boost.predict(Xtest))
ax2.set_title('Residuals vs. PERCENTPOVERTY')

fig = plt.figure()
plt.scatter(Xtest.LATITUDE, np.array(Ytest)-boost.predict(Xtest))
ax2.set_title('Residuals vs. LATITUDE')

fig = plt.figure()
plt.scatter(Xtest.SALEYEAR, np.array(Ytest)-boost.predict(Xtest))
ax2.set_title('Residuals vs. SALEYEAR')

#Forecasting examples
address = '955 9th St PA 15014'
data[(data.PROPERTYZIP == '15014') & (data.PROPERTYHOUSENUM == '1032')]
#idx = 476392 #413 Dunbar 15235: $120887 (Zestimate: $100,374)
#idx = 141398    #955 9th St PA 15014: 2015 November $120675 (Zestimate $73000, Trulia $70000)

idx = 141398

parcelID = data.ix[idx]["PARID"]
x=[]
x = data.ix[idx][['FINISHEDLIVINGAREA', 
                    'LOTAREA', 
                    'BEDROOMS', 
                    'FULLBATHS',
                    'SCHOOLCODE',
                    'CONDITION']]
                    
grade = data.ix[idx]["GRADE"]
if grade in ['A+', 'A', 'A-']: x['GRADENUM'] = 1.0
if grade == 'B+': x['GRADENUM'] = 1.7
if grade == 'B': x['GRADENUM'] = 2.0
if grade == 'B-': x['GRADENUM'] = 2.3
if grade == 'C+': x['GRADENUM'] = 2.7
if grade == 'C': x['GRADENUM'] = 3.0
if grade == 'C-': x['GRADENUM'] = 3.3
if grade in ['D+', 'D']: x['GRADENUM'] = 4.0
if grade == 'D-': x['GRADENUM'] = 4.3

x['AGE'] = 2016 - data.ix[idx]['YEARBLT']
x['SALEMONTH'] = 0
x['SALEYEAR'] = 0

idx2 = mygeodata[mygeodata.PARID == parcelID].index[0]
x = x.append(mygeodata.ix[idx2][["LATITUDE", 'LONGITUDE']])
geoID = mygeodata.ix[idx2]["GEOID"]
idx3 = tract_data[tract_data.GEOID == geoID].index[0]

x = x.append(tract_data.ix[idx3][['POPDENS',
                                  'PERCENTPOVERTY',
                                  'PERCENTBLACK',
                                  'PERCENTBACHELORS',
                                  'MED_FAM_INCOME14',
                                  'OWNEROCCRATE',
                                  'VACANCYRATE',
                                  'MEANTIMECOMMUTE']])

x = x.append(data.ix[idx][['ROOF', 'EXTERIORFINISH']])
xs = x.reshape(1,-1)

predicted_vals = [[] for i in range(3)]
for condition in range(1,4):
    for year in range(2014,2016):
        for month in range(1,13):
            x["SALEMONTH"] = month
            x["SALEYEAR"] = year
            x["CONDITION"] = condition
            predicted_vals[condition-1].append(boost.predict(xs))

x["SALEYEAR"] = 2016
for condition in range(1,4):
    for month in range(1,7):
        x["SALEMONTH"] = month
        x["CONDITION"] = condition
        predicted_vals[condition-1].append(boost.predict(xs))

fig = plt.figure()
dates = pd.date_range("1/1/2014", periods = 30, freq = "MS")
very_good, = plt.plot_date(dates, predicted_vals[0], linestyle = 'solid', color = 'blue', 
                    xdate = True, ydate = False, label = 'Excellent/Very good')
good, = plt.plot_date(dates, predicted_vals[1], linestyle = 'solid', color = 'green', 
                    xdate = True, ydate = False, label = 'Good')
average, = plt.plot_date(dates, predicted_vals[2], linestyle = 'solid', color = 'red', 
                    xdate = True, ydate = False, label = 'Average')
plt.legend(handles=[very_good, good, average])
plt.title("Market value of a property: " + address 
                                + " depending on upkeep condition")
plt.ylabel("Market value, $")

#Example: 7719 Brashear St, Pittsburgh, PA 15221 (Zestimate:~180000, Trulia ~380000)

#Get crime rate? Get foreclosure data? 
#Use k-fold cross-validation?
#Do PCA? Do k-nearest neighbors? 
#Use Zillow API? 
#Compute confidence intervals?
#Constraints on number of bedrooms?
