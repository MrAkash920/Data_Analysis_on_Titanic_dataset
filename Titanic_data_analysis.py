#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Titanic.csv')
print(df)


# In[3]:


print(df.to_string)


# In[4]:


print(df.head(10))


# In[7]:


#sort values by columan
df = df.sort_values(by='Age')
print(df.head(20))


# In[9]:


#Filter data whose Age is greater than 18
df_filter = df[df['Age'] > 18]
print (df_filter)


# In[11]:


df_filter["Age"].plot (kind = 'hist')
plt.show()


# In[12]:


#Find the age counts of all the passenger present there
age_counts = df['Age'].value_counts()
print(age_counts)


# In[13]:


# Convert age_counts to a DataFrame
age_counts_df = age_counts.reset_index()
age_counts_df.columns = ['Age', 'Count']
print(age_counts_df)


# In[17]:


#create a bar plot of age counts and age
plt.figure(figsize=(12,6))
plt.bar(age_counts_df['Age'], age_counts_df['Count'])
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.xticks(rotation=30) #used to rotate labels number
plt.show()


# In[29]:


#barplot using seaborn
import seaborn as sns
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Age', order=df['Age'].value_counts().index)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.xticks(rotation=90)
plt.show()


# In[34]:


#Crete heatmap to visualize the realtionship between age and age counts
#create a pivot table for the heatmap
data = age_counts_df.pivot_table(index = 'Age', values ='Count', aggfunc = 'sum')

plt.figure(figsize = (12,8))
sns.heatmap(data, cmap = 'coolwarm', annot = True, fmt = 'd')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age count heatmap')
plt.show()


# In[37]:


#Gender count
sex_counts = df['Sex'].value_counts()
print(sex_counts)


# In[39]:


#convert sexcounts into dataframe
sex_counts_df = sex_counts.reset_index()
sex_counts_df.columns =['Sex','Sex_Counts']
print(sex_counts_df)


# In[40]:


#create a bar plot of sex count
plt.figure(figsize=(12,6))
plt.bar(sex_counts_df['Sex'], sex_counts_df['Sex_Counts'])
plt.xlabel('Sex')
plt.ylabel('Sex Count')
plt.title('Sex Distribution')
plt.show()


# In[41]:


print(df.head(20))


# In[43]:


#Count of survived and Deceased person
survived_count = df[df['Survived']==1]['Survived'].count()
dead_count = df[df['Survived']==0]['Survived'].count()
print(f"Number of Saviors: {survived_count}")
print(f"Number of Deceased: {dead_count}")


# In[50]:


dead_count = df[df['Survived'] == 0]['Survived'].count()
survived_count = df[df['Survived'] == 1]['Survived'].count()
plt.figure(figsize=(8, 6))
plt.bar(['Dead', 'Survived'], [dead_count, survived_count], color=['red', 'green'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Survival Count (Dead vs. Survived)')
plt.show()


# In[51]:


#Count of male survived and male deceased
male_survived_count = df[(df['Survived'] == 1) & (df['Sex'] == 'male')]['Survived'].count()
male_dead_count = df[(df['Survived'] == 0) & (df['Sex'] == 'male')]['Survived'].count()

print(f"Number of Male Survivors: {male_survived_count}")
print(f"Number of Male Deceased: {male_dead_count}")


# In[54]:


plt.figure(figsize = (8,6))
plt.bar(['Dead_Male', 'Survived_Mlae'],[male_dead_count,male_survived_count],color = ['red','green'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Survival Count (Dead Male vs. Survive Male)')
plt.show()


# In[56]:


female_survived_count = df[(df['Survived'] == 1) & (df['Sex'] == 'female')]['Survived'].count()
female_dead_count = df[(df['Survived'] == 0) & (df['Sex'] == 'female')]['Survived'].count()
print(f"Number of Feamle Survivors: {female_survived_count}")
print(f"Number of Feamle Deceased: {female_dead_count}")


# In[58]:


plt.figure(figsize= (8,6))
plt.bar(['Dead_Female', 'Survived Female'],[female_dead_count,female_survived_count],color =['red','green'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Survival Count(Dead Female vs. Survived Female)')
plt.show()


# In[61]:


female_dead_count = df[(df['Survived'] == 0) & (df['Sex'] == 'female')]['Survived'].count()
male_dead_count = df[(df['Survived'] == 0) & (df['Sex'] == 'male')]['Survived'].count()
plt.figure(figsize= (8,6))
plt.bar(['Dead_Female', 'Dead Male'],[female_dead_count,male_dead_count],color =['red','red'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Dead Count(Female vs. Male)')
plt.show()


# In[62]:


female_survived_count = df[(df['Survived'] == 1) & (df['Sex'] == 'female')]['Survived'].count()
male_survived_count = df[(df['Survived'] == 1) & (df['Sex'] == 'male')]['Survived'].count()
plt.figure(figsize= (8,6))
plt.bar(['Dead_Female', 'Dead Male'],[female_survived_count,male_survived_count],color =['green','green'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Survived Count(Female vs. Male)')
plt.show()


# In[64]:


#percentage of male passenge who did not survive
male_passenger = df[df['Sex'] == 'male']
total_male_passenger = len(male_passenger)
deceased_male_passenger = len(male_passenger[male_passenger['Survived']== 0])
male_dead_percentage = (deceased_male_passenger / total_male_passenger) * 100
print(f"Percentage of Male Passengers Who Did Not Survive: {male_dead_percentage:.2f}%")


# In[65]:


#percentage of female passenge who did not survive
female_passenger = df[df['Sex'] == 'female']
total_female_passenger = len(female_passenger)
deceased_female_passenger = len(female_passenger[female_passenger['Survived']== 0])
female_dead_percentage = (deceased_female_passenger / total_female_passenger) * 100
print(f"Percentage of Male Passengers Who Did Not Survive: {female_dead_percentage:.2f}%")


# In[66]:


#claculate the mean age of passenger
mean_age = np.mean(df['Age'])
print("Mean Age: ",mean_age)


# In[72]:


#date clearing
df.dropna(inplace = True)
print(df)


# In[71]:


df.info()


# In[ ]:





# In[ ]:




