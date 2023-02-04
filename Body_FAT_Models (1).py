#!/usr/bin/env python
# coding: utf-8

# # Body_FAT Data 

# In[139]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# In[87]:


df = pd.read_csv(r"C:\Users\SHARYU\OneDrive\Desktop\good.csv")
df


# In[88]:


df.plot(kind = 'box', figsize = (20,10))


# ### After looking at the above boxplot, we conclude that age, weight, neck, chest, thigh, forearm has outliers thus need to remove it or replace it.

# In[89]:


from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats


# ## Cleaning The Variables and replacing it with median values.

# ### Age 

# In[90]:


zscore_Age = stats.zscore(df['Age'], axis = 0)
zscore_Age


# In[91]:


zscore_age = pd.DataFrame(zscore_Age,columns=['zscore_Age'])
zscore_age


# In[92]:


df = pd.concat([df,zscore_age], axis = 1)
df


# In[93]:


df.loc[df['zscore_Age']>1.96,'Age']=np.nan
df.loc[df['zscore_Age']<-1.96,'Age']=np.nan
df


# In[94]:


df['Age'] = df['Age'].fillna(df['Age'].median())
df['Age']


# ### -------------------------------------------------------------------------------

# ### Weight

# In[95]:


zscore_Weight = stats.zscore(df['Weight'], axis = 0)
zscore_Weight


# In[96]:


zscore_weight = pd.DataFrame(zscore_Weight,columns=['zscore_Weight'])
zscore_weight


# In[97]:


df = pd.concat([df,zscore_weight],axis = 1)
df 


# In[98]:


df['Weight'] = df['Weight'].fillna(df['Weight'].median())
df['Weight']


# ### -------------------------------------------------------------------

# ### Neck_Circ

# In[99]:


zscore_Neck_Circ = stats.zscore(df['Neck_Circ'], axis = 0)
zscore_Neck_Circ


# In[100]:


zscore_neck = pd.DataFrame(zscore_Neck_Circ,columns=['zscore_Neck_Circ'])
zscore_neck


# In[101]:


df = pd.concat([df,zscore_neck],axis = 1)
df


# In[102]:


df['Weight'] = df['Weight'].fillna(df['Weight'].median())
df['Weight']


# ### -------------------------------------------------------------------------------------

# ### Chest_Circ

# In[103]:


zscore_Chest_Circ = stats.zscore(df['Chest_Circ'], axis = 0)
zscore_Chest_Circ


# In[104]:


zscore_chest = pd.DataFrame(zscore_Chest_Circ,columns=['zscore_Chest_Circ'])
zscore_chest


# In[105]:


df = pd.concat([df,zscore_chest], axis = 1)
df 


# In[106]:


df['Chest_Circ'] = df['Chest_Circ'].fillna(df['Chest_Circ'].median())
df['Chest_Circ']


# ### -------------------------------------------------------------------------------------

# ### Hip_Circ

# In[107]:


df['Hip_Circ'] = df['Hip_Circ'].fillna(df['Hip_Circ'].median())
df['Hip_Circ']


# In[108]:


df['Body_FAT'] = df['Body_FAT'].fillna(df['Body_FAT'].median())
df['Body_FAT']


# In[109]:


zscore_Hip_Circ = stats.zscore(df['Hip_Circ'], axis = 0)
zscore_Hip_Circ


# In[110]:


zscore_hip = pd.DataFrame(zscore_Hip_Circ,columns=['zscore_Hip_Circ'])
zscore_hip


# In[111]:


df = pd.concat([df,zscore_hip], axis = 1)
df 


# In[112]:


df['Hip_Circ'] = df['Hip_Circ'].fillna(df['Hip_Circ'].median())
df['Hip_Circ']


# ### ----------------------------------------------------------------------------------------------

# ### Thin_Circ

# In[113]:


zscore_Thin_Circ = stats.zscore(df['Thin_Circ'])
zscore_Thin_Circ


# In[114]:


zscore_thin = pd.DataFrame(zscore_Thin_Circ, columns = ['zscore_Thin_Circ'])
zscore_thin 


# In[115]:


df = pd.concat([df,zscore_thin], axis = 1)
df 


# In[116]:


df['Thin_Circ'] = df['Thin_Circ'].fillna(df['Thin_Circ'].median())
df['Thin_Circ']


# ### -----------------------------------------------------------------

# ### Extended_Biceps_Circ

# In[117]:


zscore_Extended_Biceps = stats.zscore(df['Extended_Biceps_Circ'])
zscore_Extended_Biceps


# In[118]:


zscore_extended = pd.DataFrame(zscore_Extended_Biceps, columns = ['zscore_Extended_Biceps_Circ'])
zscore_extended


# In[119]:


df = pd.concat([df,zscore_extended], axis = 1)
df 


# In[120]:


df['Extended_Biceps_Circ'] = df['Extended_Biceps_Circ'].fillna(df['Extended_Biceps_Circ'].median())
df['Extended_Biceps_Circ']


# ### --------------------------------------------------------------------------------------

# ### Forearm_Circ

# In[121]:


zscore_Forearm_Circ = stats.zscore(df['Forearm_Circ'])
zscore_Forearm_Circ


# In[122]:


zscore_forearm = pd.DataFrame(zscore_Forearm_Circ, columns = ['zscore_Forearm_Circ'])
zscore_forearm


# In[123]:


df = pd.concat([df,zscore_forearm], axis = 1)
df 


# In[124]:


df['Forearm_Circ'] = df['Forearm_Circ'].fillna(df['Forearm_Circ'])
df['Forearm_Circ']


# ### ---------------------------------------------------------------------------------

# In[ ]:


model = ols('Body_FAT~Age+Weight+Neck_Circ+Chest_Circ+Hip_Circ+Thin_Circ+Extended_Biceps_Circ+Forearm_Circ', data = df).fit()
model1 = sm.stats.anova_lm(model)
model1
print(model.summary())
            


# ### Model - 1 with Age and Weight variables

# In[140]:


model1 = ols('Body_FAT~Age+Weight', data = df).fit()
model_1 = sm.stats.anova_lm(model1)
model_1
print(model1.summary())


# ## Model - 2 with neck, chest, hip, thigh, extended, forearm variables

# In[131]:


df.columns


# In[141]:


model2 = ols('Body_FAT~Neck_Circ+Chest_Circ+Abdomen_circ+Hip_Circ+Thin_Circ+Knee_Circ+Ankle_circ+Extended_Biceps_Circ+Forearm_Circ+Wrist_Circ', data = df).fit()
model_2 = sm.stats.anova_lm(model2)
model_2
print(model2.summary())


# ## Model 3 - with all the variables

# In[142]:


model3 = ols('Body_FAT~Age+Weight+Neck_Circ+Chest_Circ+Abdomen_circ+Hip_Circ+Thin_Circ+Knee_Circ+Ankle_circ+Extended_Biceps_Circ+Forearm_Circ+Wrist_Circ', data = df).fit()
model_3 = sm.stats.anova_lm(model3)
model_3
print(model.summary())  


# ### ----------------------------------------------------------------------------------------------------------------

# ## AIC & BIC Values

# In[143]:


print(model1.aic)


# In[144]:


print(model2.aic)


# In[145]:


print(model3.aic)


# ### According to above observation model 3(with all the variables) is the good fit. 

# ## --------------------------------------------------------------------------------------------------------

# ## Durbin-Watson_Test

# In[146]:


from statsmodels.stats.stattools import durbin_watson


# In[148]:


durbin_watson(model_1)


# In[150]:


durbin_watson(model_2)


# In[151]:


durbin_watson(model_3)


# # ++++++++++++++++++++++++++++++++++++++++++++++++++

# In[ ]:




