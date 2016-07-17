import pandas as pd

income = pd.read_csv('C:\\Users\\alex314\\Desktop\\Allegheny\\ACS_14_5YR_S1903_with_ann.csv',
                        dtype={"GEO.id2" : "str", "HC01_EST_VC02": "float"}) 
myincome = income[["GEO.id2", "HC01_EST_VC02", "HC02_EST_VC02",
                    "HC02_EST_VC22"]]
myincome = myincome.rename(columns = {"GEO.id2" : "GEOID", 
                                      "HC01_EST_VC02" : "NUMHOUSEHOLDS", 
                                      "HC02_EST_VC02": "INCOME",
                                      "HC02_EST_VC22": "MED_FAM_INCOME14" 
                                      #"HC02_MOE_VC02" : "MOE_INCOME"
                                      })
myincome = myincome.drop(myincome[myincome["INCOME"]=="-"].index)
myincome = myincome.drop(myincome[myincome["MED_FAM_INCOME14"]=="-"].index)
myincome["INCOME"]=pd.Series([float(myincome.INCOME[idx]) 
                            for idx in myincome.index], index=myincome.index)
myincome["MED_FAM_INCOME14"]=pd.Series([float(myincome.MED_FAM_INCOME14[idx]) 
                            for idx in myincome.index], index=myincome.index)

education = pd.read_csv("C:\\Users\\alex314\\Desktop\\Allegheny\\ACS_14_5YR_S1501.csv", 
                        dtype={"GEO.id2" : "str"}) 
education = education[["GEO.id2", "HC01_EST_VC17"]]
education = education.rename(columns = {"GEO.id2" : "GEOID", 
                                      #"HC01_EST_VC16" : "PERCENTHS", 
                                      "HC01_EST_VC17": "PERCENTBACHELORS"})
education = education.drop(education[education["PERCENTBACHELORS"]=="-"].index)
education["PERCENTBACHELORS"] = \
    [float(education["PERCENTBACHELORS"][idx]) for idx in education.index]

df1 = pd.merge(education, myincome, how = 'inner', on = 'GEOID')                             

race = pd.read_csv("C:\\Users\\alex314\\Desktop\\Allegheny\\ACS_14_5YR_B02001.csv", 
                        dtype={"GEO.id2" : "str", "HD01_VD01" : "float", 
                                                    "HD01_VD03" : "float"})
myrace = race[["GEO.id2", "HD01_VD01", "HD01_VD03"]]
myrace = myrace.rename(columns = {"GEO.id2" : "GEOID", 
                                  "HD01_VD01" : "TOTALPOP", 
                                  "HD01_VD03" : "BLACKPOP"})
myrace["PERCENTBLACK"] = myrace["BLACKPOP"] / myrace["TOTALPOP"] * 100
df2 = pd.merge(myrace, df1, how = 'inner', on = 'GEOID')         

census10 = pd.read_csv("C:\\Users\\alex314\\Desktop\\Allegheny\\Allegheny_Census_2013.txt",
                    dtype={"GEOID10" : "str"}) 
mycensus = census10[["GEOID10", "ALAND10"]]
mycensus = mycensus.rename(columns = {'GEOID10': 'GEOID', 'ALAND10' : 'AREA'})
df3 = pd.merge(df2, mycensus, how='left', on='GEOID')
df3["POPDENS"] = df3["TOTALPOP"] / df3["AREA"] * 1000000
 
poverty = pd.read_csv("C:\\Users\\alex314\\Desktop\\Allegheny\\ACS_14_5YR_S1701.csv",
                        dtype={"GEO.id2" : "str"})
poverty = poverty[["GEO.id2", "HC03_EST_VC01"]] 
poverty = poverty.rename(columns = {"GEO.id2" : "GEOID", 
                                  "HC03_EST_VC01" : "PERCENTPOVERTY"})
poverty = poverty.drop(poverty[poverty["PERCENTPOVERTY"]=="-"].index)
poverty["PERCENTPOVERTY"] = \
    [float(poverty["PERCENTPOVERTY"][idx]) for idx in poverty.index]

df4 = pd.merge(df3, poverty, how='left', on='GEOID')       

owner = pd.read_csv("C:\\Users\\alex314\\Desktop\\Allegheny\\ACS_14_5YR_DP04.csv", 
                        dtype={"GEO.id2" : "str"})
myowner = owner[["GEO.id2", "HC03_VC64", "HC03_VC05"]] 
myowner = myowner.rename(columns = {"GEO.id2" : "GEOID", 
                                  "HC03_VC64" : "OWNEROCCRATE",
                                  "HC03_VC05" : "VACANCYRATE"})
myowner = myowner.drop(myowner[myowner["OWNEROCCRATE"]=="-"].index)
myowner = myowner.drop(myowner[myowner["VACANCYRATE"]=="-"].index)
myowner["OWNEROCCRATE"] = \
    [float(myowner["OWNEROCCRATE"][idx]) for idx in myowner.index]
myowner["VACANCYRATE"] = \
    [float(myowner["VACANCYRATE"][idx]) for idx in myowner.index]
df5 = pd.merge(myowner, df4, how = 'inner', on = 'GEOID')   

commute = pd.read_csv("C:\\Users\\alex314\\Desktop\\Allegheny\\ACS_14_5YR_DP03.csv", 
                        dtype={"GEO.id2" : "str"})
mycommute = commute[["GEO.id2", "HC01_VC36"]] 
mycommute = mycommute.rename(columns = {"GEO.id2" : "GEOID", 
                                  "HC01_VC36" : "MEANTIMECOMMUTE"})
mycommute = mycommute.drop(mycommute[mycommute["MEANTIMECOMMUTE"]=="-"].index)
mycommute = mycommute.drop(mycommute[mycommute["MEANTIMECOMMUTE"]=="N"].index)
mycommute["MEANTIMECOMMUTE"] = \
    [float(mycommute["MEANTIMECOMMUTE"][idx]) for idx in mycommute.index]
df6 = pd.merge(mycommute, df5, how = 'inner', on = 'GEOID')   

df6.to_csv('C:\\Users\\alex314\\Desktop\\Allegheny\\tract_data.csv')
                        
                        