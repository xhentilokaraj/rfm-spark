#!/usr/bin/env python
# coding: utf-8

# In[72]:


import findspark
findspark.init()

import pandas as pd

get_ipython().run_line_magic('load_ext', 'autotime')

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import pyspark

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *


# In[4]:


spark = SparkSession     .builder     .appName("RFM Analysis with PySpark")     .getOrCreate()


# In[5]:


spark


# Data Source: https://www.kaggle.com/carrie1/ecommerce-data

# In[6]:


data = spark.read.format("csv").option("header", "true").load("data.csv")


# In[7]:


data


# In[8]:


data.columns


# In[9]:


data.printSchema()


# In[10]:


# cache data in memory
data.cache().count()


# In[33]:


data.show(5)


# # 1 Data Pre-Processing

# In[13]:


data = data.withColumn("Quantity", data["Quantity"].cast(IntegerType()))
data = data.withColumn("UnitPrice", data["UnitPrice"].cast(DoubleType()))


# In[14]:


# define Total column
data = data.withColumn("Total", round(data["UnitPrice"] * data["Quantity"], 2))


# In[15]:


# change date format
data = data.withColumn("Date", to_date(unix_timestamp("InvoiceDate", "MM/dd/yyyy").cast("timestamp")))


# In[16]:


# calculate difference in days between 2011-12-31 and the Invoice Date
data = data.withColumn("RecencyDays", expr("datediff('2011-12-31', Date)"))


# # 2 Create RFM Table

# In[50]:


# Creation of RFM table

rfm_table = data.groupBy("CustomerId")                        .agg(min("RecencyDays").alias("Recency"),                              count("InvoiceNo").alias("Frequency"),                              sum("Total").alias("Monetary"))


# In[51]:


rfm_table = rfm_table.withColumn("Monetary", round(rfm_table["Monetary"], 2))


# In[52]:


rfm_table.printSchema()


# In[54]:


rfm_table.show(5)


# In[20]:


rfm_table.cache().count()


# # 3 Computing Quartiles of RFM values

# In[56]:


r_quartile = rfm_table.approxQuantile("Recency", [0.25, 0.5, 0.75], 0)
f_quartile = rfm_table.approxQuantile("Frequency", [0.25, 0.5, 0.75], 0)
m_quartile = rfm_table.approxQuantile("Monetary", [0.25, 0.5, 0.75], 0)


# In[57]:


# calculate Recency based on quartile

rfm_table = rfm_table.withColumn("R_Quartile",                                  when(col("Recency") >= r_quartile[2] , 1).                                 when(col("Recency") >= r_quartile[1] , 2).                                 when(col("Recency") >= r_quartile[0] , 3).                                 otherwise(4))


# In[58]:


# calculate Frequency based on quartile

rfm_table = rfm_table.withColumn("F_Quartile",                                  when(col("Frequency") > f_quartile[2] , 4).                                 when(col("Frequency") > f_quartile[1] , 3).                                 when(col("Frequency") > f_quartile[0] , 2).                                 otherwise(1))


# In[59]:


# calculate Monetary based on quartile

rfm_table = rfm_table.withColumn("M_Quartile",                                  when(col("Monetary") >= m_quartile[2] , 4).                                 when(col("Monetary") >= m_quartile[1] , 3).                                 when(col("Monetary") >= m_quartile[0] , 2).                                 otherwise(1))


# In[60]:


# combine the scores (R_Quartile, F_Quartile,M_Quartile) together.

rfm_table = rfm_table.withColumn("RFM_Score", concat(col("R_Quartile"), col("F_Quartile"), col("M_Quartile")))


# In[62]:


rfm_table.show(10)


# # 4 RFM Analysis

# In[63]:


# Best customers

rfm_table.select("CustomerID").where("RFM_Score == 444").show(10)


# In[64]:


# group by RFM Score

grouped_by_rfmscore = rfm_table.groupBy("RFM_Score").count().orderBy("count", ascending=False)


# In[65]:


# convert Spark dataframe to pandas in order to visualize data

grouped_by_rfmscore_pandas = grouped_by_rfmscore.toPandas()


# In[66]:


grouped_by_rfmscore_pandas


# In[67]:


grouped_by_rfmscore_pandas['RFM_Score'] = "Seg " + grouped_by_rfmscore_pandas['RFM_Score'].map(str)


# In[73]:


data = [go.Bar(x=grouped_by_rfmscore_pandas['RFM_Score'], y=grouped_by_rfmscore_pandas['count'])]

layout = go.Layout(
    title=go.layout.Title(
        text='Customer RFM Segments'
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text='RFM Segment'
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text='Number of Customers'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='rfm_Segments')


# In[ ]:




