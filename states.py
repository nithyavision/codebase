# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:59:01 2023

@author: karth
"""


import pandas as pd
import datetime
import numpy as np
#from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score




st.title("eda")



df= pd.read_csv("State_wise_Health_income (1).csv")  # read a CSV file inside the 'data" folder next to 'app.py'
df_numb = df.select_dtypes(include=[np.number])


tab1, tab2, tab3,tab4,tab5= st.tabs(["Univariate", "Bivariate","Multivariate" ,"unique counts","Boxplots"])


with tab1:
   st.header("Univariate Analysis")
   
   fig = plt.figure(figsize=(9, 7))
   sns.histplot(data = df_numb)
   st.pyplot(fig)
       
   
 
   st.title("Scatter plot : Per capital income")
   fig, ax = plt.subplots(figsize=(7,5))
   ax.scatter(df.index,df["Per_capita_income"])
   st.pyplot(fig)
   
   st.title("Histogram : GDP")
   fig, ax = plt.subplots()
   ax.hist(df["GDP"], bins=20)
   st.pyplot(fig)
   
   st.title("Histogram : States")
   fig, ax = plt.subplots()
   ax.hist(df["States"], bins=20)
   st.pyplot(fig)
   
  
   st.title("Barchart : Health indices 1")
   st.bar_chart(df["Health_indeces1"])
   st.title("Barchart : Health indices 2")
   st.bar_chart(df["Health_indices2"])
   
   
   st.area_chart(df["States"])
  # plt.scatter(df.index,df["Per_capita_income"])
   #plt.show()
  #sns.set(style='whitegrid')
   #seaborn.scatterplot(x=A, y="signal", data=fmri)
   
   
   
with tab2:
   st.write("Bivariate Analysis")
   
   st.write("Pairplot")
   sns.set_style("whitegrid")
   #fig = plt.figure(figsize=(9, 7))
  # fig=sns.pairplot(data=df,diag_kind="kde",)
   fig=sns.pairplot(data = df, kind='reg', diag_kind = 'kde')
   st.pyplot(fig)
   
   
   st.write("Heatmap")
   fig = plt.figure(figsize=(10, 4))
   #sns.pairplot(data = df, kind='reg', diag_kind = 'kde')
   sns.heatmap(df.corr(),annot=True)
   st.pyplot(fig)
   
   
  
   
with tab3:
   st.header("Multivariate Analysis")
   #scaling
   df.drop(["States","Unnamed: 0"], axis=1,inplace=True) 
   plt.style.use('ggplot')
   ss = StandardScaler()
   df_scaled = pd.DataFrame(ss.fit_transform(df),columns = df.columns)
   df_scaled.describe()
   X=df_scaled
   Y=df_scaled
   st.write("Before scaling")
   st.write(df.describe())
   st.write("After scaling")
   st.write(df_scaled.describe())
   wss =[] 
   for i in range(1,11):
       KM = KMeans(n_clusters=i,random_state=1)
       KM.fit(Y)
       wss.append(KM.inertia_)
       labels = KM.predict(Y)
   st.write(wss)
   
with tab4:
   st.write("First and last five data sets")
   st.table(df.head(5))
   st.table(df.tail(5))
   
   
   st.write("Shape of the data frame : ") 
   st.write(df.shape)
   
   
   st.write("checking for nan values ") 
   nan_df = df[df.isna().any(axis=1)]
   st.write(nan_df.head())
   
   df=df.dropna()
   
   st.write("checking for null values ") 
   st.write(df.isnull().sum())
   
   
   st.write("Checking for Duplicates ") 
   n_duplicates = df.duplicated().sum()
   st.write(f"Seem to have {n_duplicates} duplicates in database.")
   
   
   
   
   st.write("Describe ") 
   st.write(df.describe(include="all").T)
   st.header("Unique Counts")
   unique_counts = []
   for col in df.columns:
       unique_counts.append((col,df[col].nunique()))
   unique_counts = sorted(unique_counts, key=lambda x: x[1],reverse = True)
   
   st.write("No of unique values in each column")
   for col,nunique in unique_counts:
       st.write(f"{col}: {nunique}")
       
   st.write("Outlier Analysis")
   dfo=pd.DataFrame()
   dfo= df.select_dtypes(exclude = 'object')
   Q1 = dfo.quantile(0.25)
   Q3 = dfo.quantile(0.75)
   IQR = Q3 - Q1
   ans = ((dfo < (Q1 - 1.5 * IQR)) | (dfo> (Q3 + 1.5 * IQR))).sum()
   dfo = pd.DataFrame(ans).reset_index().rename(columns = {'index':'column', 0:'count_of_outliers'})
   st.write(dfo)
 
    
    
with tab5:
    
    st.write("Outlier Analysis")
    dfo=pd.DataFrame()
    dfo= df.select_dtypes(exclude = 'object')
    Q1 = dfo.quantile(0.25)
    Q3 = dfo.quantile(0.75)
    IQR = Q3 - Q1
    ans = ((dfo < (Q1 - 1.5 * IQR)) | (dfo> (Q3 + 1.5 * IQR))).sum()
    dfo = pd.DataFrame(ans).reset_index().rename(columns = {'index':'column', 0:'count_of_outliers'})
    st.write(dfo)
    
    for column in df_numb:
       fig = plt.figure(figsize=(9, 7))
       sns.boxplot(data=df_numb,x=column)
       st.pyplot(fig)
       
       
    st.write(df.loc[df['Health_indeces1'] == -10])
       
