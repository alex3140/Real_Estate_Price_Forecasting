import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import cross_validation
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor

data = pd.read_csv('C:\\Users\\alex314\\Desktop\\Allegheny\\Allegheny_data.csv', 
   low_memory = False, dtype = {'PROPERTYZIP': 'str', 'PROPERTYHOUSENUM' : 'str'})

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
data1 = data[(data["SALEDESC"].isin(["VALID SALE", "OTHER VALID", "SALE NOT ANALYZ"]))
            & (data["USEDESC"].isin(["SINGLE FAMILY", "TOWNHOUSE", "ROWHOUSE"]))
            & (data["CONDITION"].isin([1,2,3,4,5,7]))
            & (data["CDUDESC"]).isin(["EXCELLENT", "VERY GOOD", "GOOD", 
                                    "AVERAGE", "FAIR", "POOR"])]
                                             
saledates = pd.Series([datetime.strptime(data1.SALEDATE[idx], "%m-%d-%Y")
                        for idx in data1.index], index = data1.index)
bad_idx=[]
for idx in saledates.index:
    date=saledates[idx]
    if date.year < 1908:
        print date.year, idx
        bad_idx.append(idx)
data1 = data1.drop(bad_idx)

data1["SALEDATE"] = pd.Series(saledates, index=data1.index)
data1["SALEYEAR"] = pd.Series([data1.SALEDATE[idx].year for idx in data1.index],
                                index = data1.index)
data1["SALEMONTH"] = pd.Series([data1.SALEDATE[idx].month for idx in data1.index],
                                index = data1.index)

data1 = data1[data1.PROPERTYHOUSENUM.notnull()] 
data1["ADDRESS"] = pd.Series(
                    [data1.PROPERTYHOUSENUM[idx] + " "
                    + str(data1.PROPERTYADDRESS[idx]) + " PA " 
                    + data1.PROPERTYZIP[idx]
                    for idx in data1.index], index = data1.index)
                    
data2 = data1[['PARID', 'ADDRESS', 
                'PROPERTYZIP', 'SCHOOLCODE', 
                'LOTAREA', 'SALEDATE', 
                'SALEYEAR', 'SALEMONTH',
                'SALEPRICE', 'PREVSALEDATE', 
                'PREVSALEPRICE', 'FAIRMARKETTOTAL', 
                'YEARBLT', 'ROOF', 
                'EXTERIORFINISH', 'GRADE', 
                'CONDITION', 'BEDROOMS', 
                'FULLBATHS', 'HALFBATHS',  
                'FIREPLACES', 'BASEMENT', 
                'BSMTGARAGE', 'FINISHEDLIVINGAREA',
                'SCHOOLRANK', 'SALEDESC', 'TYPE']]
                    
data3 = data2[data2.GRADE.notnull()]
geodata = pd.read_csv(
 'C:\\Users\\alex314\\Desktop\\Allegheny\\April2016CentroidBlock.csv', 
                    low_memory = False)
#In this dataset the labels for Lat and Long are flipped
mygeodata = geodata[["PIN", "Latitude", "Longitude", "geo_id_tra"]]
mygeodata = mygeodata.rename(columns = {"PIN":"PARID", "Latitude" : "LONGITUDE", 
                             "Longitude" : "LATITUDE", "geo_id_tra" : "GEOID"})
data4 = pd.merge(data3, mygeodata, how = 'left', on = 'PARID')

data4["AGE"] = data4.SALEYEAR - data4.YEARBLT
data4.CONDITION[data4.CONDITION == 7.0] = 1.0

#XX,X,A+,A,A- are custom-designed or historic homes and should be ignored
data4["GRADENUM"] = 0
data4["GRADENUM"][data4.GRADE == "B+"] = 1.7
data4["GRADENUM"][data4.GRADE == "B"] = 2.0
data4["GRADENUM"][data4.GRADE == "B-"] = 2.3
data4["GRADENUM"][data4.GRADE == "C+"] = 2.7
data4["GRADENUM"][data4.GRADE == "C"] = 3.0
data4["GRADENUM"][data4.GRADE == "C-"] = 3.3
data4["GRADENUM"][data4.GRADE == "D+"] = 3.7
data4["GRADENUM"][data4.GRADE == "D"] = 4.0
data4["GRADENUM"][data4.GRADE == "D-"] = 4.3

tract_data = pd.read_csv(
    'C:\\Users\\alex314\\Desktop\\Allegheny\\tract_data.csv',
    dtype={"GEOID" : "str"})
tract_data = tract_data.drop("Unnamed: 0", axis = 1)
data5 = pd.merge(data4, tract_data, how = "left", on = "GEOID")

data6 = data5[data5["SALEYEAR"].isin(range(2008,2017))]
data6 = data6[~((data6.SALEYEAR == 2016) & (data6.SALEMONTH.isin(range(7,13))))]

modeldata = data6[(data6.SALEPRICE < 360000)
                    & (data6.SALEPRICE > 10000)
                    #& (data6.FAIRMARKETTOTAL > 15000)
                    & (data6.LOTAREA < 21000) 
                    & (data6.BEDROOMS > 0)
                    & (data6.BEDROOMS < 7)
                    & (data6.FULLBATHS > 0)
                    & (data6.FULLBATHS < 5)
                    & (data6.FINISHEDLIVINGAREA < 3500)
                    & (2.7 * data6.FAIRMARKETTOTAL > data6.SALEPRICE)
                    & (0.8 * data6.FAIRMARKETTOTAL < data6.SALEPRICE)
                    & (data6. AGE >= 0)
                    & (data6.AGE < 130)
                    & (data6.GRADENUM > 1)]
                    #& (data6.SALEDESC.isin(["VALID SALE", "OTHER VALID"]))]

modeldata["SCHOOLRANK"][modeldata.SCHOOLRANK == 1.3] = 1
modeldata["SCHOOLRANK"][modeldata.SCHOOLRANK == 3.3] = 3

obj = modeldata.groupby(["SALEYEAR","SALEMONTH"]).SALEPRICE
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
dates = pd.date_range("1/1/2008", periods = obj.mean().shape[0], freq="MS")
median, = plt.plot_date(dates, obj.median(), 'b-', tz = None, 
                    xdate = True, ydate = False, label = 'Median Price')
mean, = plt.plot_date(dates, obj.mean(), 'r-', label = 'Mean Price')
plt.legend(handles=[median, mean])
plt.title("Mean and Median Sale Price per Month, $")
ax = fig.add_subplot(2,1,2)
median, = plt.plot_date(dates, obj.count(), 'b-', tz = None, 
                    xdate = True, ydate = False)
plt.title("Number of Sales per Month")

#Boxplots
plt.figure()
sns.boxplot(x = modeldata.FINISHEDLIVINGAREA)
plt.figure()
sns.boxplot(x = modeldata.LOTAREA)
plt.figure()
sns.boxplot(x = data4.BASEMENT, y = data4.SALEPRICE)
plt.figure()
sns.boxplot(x = data4.EXTERIORFINISH, y = data4.SALEPRICE)
plt.figure()
sns.boxplot(x = modeldata.SALEYEAR, y = modeldata.SALEPRICE)
plt.figure()
sns.boxplot(x = data.SCHOOLRANK, y = data6.SALEPRICE)
plt.figure()
sns.boxplot(x = modeldata.CONDITION, y = modeldata.SALEPRICE)
g = sns.PairGrid(modeldata, vars = ['SALEPRICE', 'FINISHEDLIVINGAREA'])
g = g.map_diag(plt.hist, edgecolor = "w")
g = g.map_offdiag(plt.scatter, edgecolor = "w", s = 15)
                  
#Map
zipcode_list = ['15202', '15209', '15212', '15214', '15229', '15116', '15206',
              '15235', '15146', '15221', '15208', '15218', '15217', '15207',
              '15120', '15227', '15226', '15234', '15210', '15216', '15211',
               '15220', '15205', '15204', '15136']               
zipdata = modeldata[modeldata.PROPERTYZIP.isin(zipcode_list)]

fig = plt.figure()
plt.scatter(modeldata.LONGITUDE, modeldata.LATITUDE)
#plt.scatter(zipdata.LONGITUDE, zipdata.LATITUDE, color='r')
plt.title("Property locations in Allegheny County")
plt.ylabel("Latitude")
plt.xlabel("Longitude")
                                      
#Model Building
features = ['FINISHEDLIVINGAREA', 'LOTAREA', 
            'BEDROOMS', 'FULLBATHS',
            'SCHOOLCODE',  'SCHOOLRANK',
            'TYPE', 'ROOF', 'EXTERIORFINISH', 'BASEMENT',
            'CONDITION','GRADENUM', 
            'AGE', 'SALEMONTH',
            'SALEYEAR', 'LATITUDE',
            'LONGITUDE', 'POPDENS',
            'PERCENTPOVERTY', #'PERCENTBLACK',
            'PERCENTBACHELORS', 'MED_FAM_INCOME14',
            'OWNEROCCRATE', 'VACANCYRATE',
            'MEANTIMECOMMUTE']
                                  
boostdata = modeldata[['SALEPRICE']+features].dropna()
X, Y = boostdata[features], boostdata['SALEPRICE']
Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(
        X, Y, test_size=.1, random_state = 555)
        
#Model 1: Gradient tree boosting model          
boost = GradientBoostingRegressor(learning_rate = .03, subsample = .5,
                         n_estimators = 1900, max_features = "log2",
                         max_depth = 7, min_samples_leaf=3)                
                                                              
boost.fit(Xtrain, Ytrain)
RMSEs_boost = [mean_squared_error(Ytrain, boost.predict(Xtrain))**.5,
               mean_squared_error(Ytest, boost.predict(Xtest))**.5]
APE = abs(Ytest - boost.predict(Xtest)) / Ytest
MedAPE = APE.median()
MeanAPE = APE.mean() 
percent5 = sum(APE < .05) / float(Ytest.shape[0])
percent10 = sum(APE < .1) / float(Ytest.shape[0])
percent15 = sum(APE < .15) / float(Ytest.shape[0])

print "***(Stochastic) Gradient Boosting Tree Model***"
print "Median Absolute Percent Error: {} %".format((round(MedAPE,4) * 100))
print "Mean Absolute Percent Error: {} %".format(round(MeanAPE, 5) * 100)
print "Percent of predictions with error below 5%: {}%".format(
                                                      round(percent5, 4) * 100)
print "Percent of predictions with error below 10%: {}%".format(
                                                      round(percent10, 4) * 100)
print "Percent of predictions with error below 15%: {}%".format(
                                                      round(percent15, 4) * 100)
print "Root MSE on train set: %s, Root MSE on test set: %s" \
                            %(round(RMSEs_boost[0], 1),round(RMSEs_boost[1], 1))
print "R-square is {} %".format(
                        (round(r2_score(Ytest,boost.predict(Xtest)), 4) * 100))

#Feature importance plot
featImportances = boost.feature_importances_
pos = np.arange(len(features))
pairs = zip(features, featImportances)
sorted_pairs = sorted(pairs, key = lambda pair: pair[1])
features_sorted, featImportances_sorted = zip(*sorted_pairs)
fig, ax = plt.subplots()
plt.barh(pos, featImportances_sorted, 1, color = "blue")
plt.yticks(pos,features_sorted)
ax.set_title('Gradient Boosting: Relative Feature Importance')

#Goodness-of-fit plot
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.scatter(np.array(Ytest), boost.predict(Xtest))
ax.plot(np.arange(modeldata.SALEPRICE.min(),modeldata.SALEPRICE.max(), .1), 
 np.arange(modeldata.SALEPRICE.min(),modeldata.SALEPRICE.max(),.1), color='navy')
ax.set_title('Predicted price vs. Actual price')
ax.set_ylabel('Predicted price, $')
ax2 = fig.add_subplot(2, 1, 2)
ax2.scatter(np.array(Ytest), np.array(Ytest)-boost.predict(Xtest))
ax2.set_title('Residuals vs. actual price')
ax2.set_xlabel('Actual price, $')
ax2.set_ylabel('Residual, $')

#Study residuals and unusual predicted values
outliers = Xtest[APE>.6]
plt.figure()
plt.scatter(Xtest.LONGITUDE, Xtest.LATITUDE, c = APE, cmap = plt.cm.OrRd)
#plt.scatter(outliers.LONGITUDE, outliers.LATITUDE, color='blue')
plt.title("Color coded map of absolute percentage error on test samples")
plt.ylabel("Latitude")
plt.xlabel("Longitude")

forecasted_price = pd.Series(boost.predict(Xtest.ix[outliers.index]), 
                                                    index=outliers.index)
pd.concat([Xtest.ix[outliers.index], modeldata.ix[outliers.index][[
    'ADDRESS','SALEPRICE', 'FAIRMARKETTOTAL','SALEDESC']], forecasted_price], 
                            axis = 1).drop (["LATITUDE", "LONGITUDE"], axis=1)

#Study various factors
fig = plt.figure()
plt.scatter(Xtest.AGE, np.array(Ytest)-boost.predict(Xtest))
ax2.set_title('Residuals vs. age')

fig = plt.figure()
sns.boxplot(x = modeldata.SALEYEAR, y = np.array(Ytest)-boost.predict(Xtest))

#Prediction intervals (70%)
#boost_upper = GradientBoostingRegressor(loss='quantile',alpha=.85, 
#    learning_rate=.03, subsample=.5, n_estimators = 2000, max_features = "sqrt", max_depth = 7)
#boost_upper.fit(Xtrain, Ytrain)

#boost_lower = GradientBoostingRegressor(loss='quantile', alpha=.15, 
#    learning_rate=.03, subsample=.5, n_estimators = 2000, 
#                        max_features = "sqrt", max_depth = 7)
#boost_lower.fit(Xtrain, Ytrain)

#Forecasting examples
#idx = 476445 413 Dunbar PA 15235: (Zestimate: $100,374)
#335 37TH ST PA 15201 Sold for 155K, FAIRMARKETTOTAL ~37K

#Train the algorithm on X
#boost.fit(X, Y)

data[(data.PROPERTYZIP == '15136') & (data.PROPERTYHOUSENUM == '834')]
address = '834 Russellwood Ave, Mc Kees Rocks, PA 15136'
idx = 461365 
print address
print "CONDITION: {}, GRADE: {}".format(data.ix[idx].CONDITIONDESC, 
                                                            data.ix[idx].GRADE)
print "FAIRMARKETTOTAL: {}, SALEPRICE:{}, SALEYEAR: {}".format(
    data.ix[idx].FAIRMARKETTOTAL, data.ix[idx].SALEPRICE, data.ix[idx].SALEDATE)
parcelID = data.ix[idx]["PARID"]
x = data.ix[idx][['FINISHEDLIVINGAREA', 'LOTAREA', 
                    'BEDROOMS', 'FULLBATHS',
                    'SCHOOLCODE', 'SCHOOLRANK', 
                    'TYPE', 'ROOF', 'EXTERIORFINISH', 'BASEMENT',
                    'CONDITION']]
                    
grade = data.ix[idx]["GRADE"]
if grade == 'B+': x['GRADENUM'] = 1.7
if grade == 'B': x['GRADENUM'] = 2.0
if grade == 'B-': x['GRADENUM'] = 2.3
if grade == 'C+': x['GRADENUM'] = 2.7
if grade == 'C': x['GRADENUM'] = 3.0
if grade == 'C-': x['GRADENUM'] = 3.3
if grade in ['D+', 'D']: x['GRADENUM'] = 4.0
if grade == 'D-': x['GRADENUM'] = 4.3

x['AGE'] = 0
x['SALEMONTH'] = 0
x['SALEYEAR'] = 0

idx2 = mygeodata[mygeodata.PARID == parcelID].index[0]
x = x.append(mygeodata.ix[idx2][["LATITUDE", 'LONGITUDE']])
geoID = mygeodata.ix[idx2]["GEOID"]
idx3 = tract_data[tract_data.GEOID == geoID].index[0]

x = x.append(tract_data.ix[idx3][['POPDENS',
                                  'PERCENTPOVERTY',
                                  #'PERCENTBLACK',
                                  'PERCENTBACHELORS',
                                  'MED_FAM_INCOME14',
                                  'OWNEROCCRATE',
                                  'VACANCYRATE',
                                  'MEANTIMECOMMUTE']])
xs = x.reshape(1,-1)

predicted_vals = [[] for i in range(5)]
#lower_bounds, upper_bounds = [[] for i in range(4)], [[] for i in range(5)]
for condition in range(1,6):
    for year in range(2013,2017):
        x['AGE'] = year - data.ix[idx]['YEARBLT']
        for month in range(1,13):
            x["SALEMONTH"] = month
            x["SALEYEAR"] = year
            x["CONDITION"] = condition
            predicted_vals[condition-1].append(boost.predict(xs)[0])
            #lower_bounds[condition-1].append(boost_lower.predict(xs))
            #upper_bounds[condition-1].append(boost_upper.predict(xs))

plt.figure()
dates = pd.date_range("1/1/2013", periods = 48, freq = "MS").values
#very_good, = plt.plot(dates, predicted_vals[0], linestyle = 'solid', 
#    color = 'blue', marker="o", label = 'Excellent/Very good')
good, = plt.plot(dates, predicted_vals[1], linestyle = 'solid', 
    color = 'blue', marker="o",label = 'Good')
fair, = plt.plot(dates, predicted_vals[3], linestyle = 'solid', 
                    marker="o", color = 'orange', label = 'Fair')                    
average, = plt.plot(dates, predicted_vals[2], linestyle = 'solid', 
    color = 'green', marker='o', label = 'Average')
#poor, = plt.plot(dates, predicted_vals[4], linestyle = 'solid', 
#    color = 'orange', marker='o', label = 'Average')
good_and_average, = plt.plot(dates, 
    (np.array(predicted_vals[1]) + np.array(predicted_vals[2]))/2, 
    linestyle = 'dashed',  marker=None, color = 'green', label = 'Good/Average')
fair_and_average, = plt.plot(dates, 
    (np.array(predicted_vals[3]) + np.array(predicted_vals[2]))/2, 
    linestyle = 'dashed',  marker=None, color = 'green', label = 'Fair/Average')
plt.fill_between(dates, (np.array(predicted_vals[1]) + np.array(predicted_vals[2]))/2, 
    (np.array(predicted_vals[3]) + np.array(predicted_vals[2]))/2, 
    facecolor='green', alpha=0.2, interpolate=True)
plt.fill_between(dates, (np.array(predicted_vals[1]) + np.array(predicted_vals[0]))/2, 
    (np.array(predicted_vals[1]) + np.array(predicted_vals[2]))/2, 
    facecolor='blue', alpha=0.2, interpolate=True)
plt.fill_between(dates, (np.array(predicted_vals[2]) + np.array(predicted_vals[3]))/2, 
   (np.array(predicted_vals[3]) + np.array(predicted_vals[4]))/2, 
   facecolor='orange', alpha=0.2, interpolate=True)
plt.legend(handles=[good, average, fair], bbox_to_anchor=(1, .1))
plt.title("Market value of a property: " + address )
plt.ylabel("Market value, $")
