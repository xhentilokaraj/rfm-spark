#!/usr/bin/env python
# coding: utf-8

# In[461]:


import findspark
findspark.init()

import pandas as pd

get_ipython().run_line_magic('load_ext', 'autotime')

import chart_studio.plotly as py
init_notebook_mode(connected=True)

import pyspark

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *


# In[6]:


spark = SparkSession     .builder     .appName("RFM Analysis with PySpark")     .getOrCreate()


# In[7]:


spark


# Data Source: https://www.kaggle.com/carrie1/ecommerce-data

# In[232]:


data = spark.read.format("csv").option("header", "true").load("data.csv")


# In[233]:


data


# In[234]:


data.columns


# In[235]:


data.printSchema()


# In[237]:


# cache data in memory
data.cache().count()


# In[432]:


data.groupBy("RecencyDays").count().show()


# In[426]:


data.show()


# # 1 Data Pre-Processing

# In[239]:


data = data.withColumn("Quantity", data["Quantity"].cast(IntegerType()))
data = data.withColumn("UnitPrice", data["UnitPrice"].cast(DoubleType()))


# In[240]:


# define Total column
data = data.withColumn("Total", round(data["UnitPrice"] * data["Quantity"], 2))


# In[241]:


# change date format
data = data.withColumn("Date", to_date(unix_timestamp("InvoiceDate", "MM/dd/yyyy").cast("timestamp")))


# In[416]:


# calculate difference in days between 2011-12-31 and the Invoice Date
data = data.withColumn("RecencyDays", expr("datediff('2011-12-31', Date)"))


# # 2 Create RFM Table

# In[436]:


# Creation of RFM table

rfm_table = data.groupBy("CustomerId")                        .agg(max("Date").alias("LastPurchase"),                              min("RecencyDays").alias("Recency"),                              count("InvoiceNo").alias("Frequency"),                              sum("Total").alias("Monetary"))


# In[437]:


rfm_table.printSchema()


# In[438]:


rfm_table.show(5)


# In[441]:


rfm_table.cache().count()


# # 3 Computing Quartiles of RFM values

# In[442]:


r_quartile = rfm_table.approxQuantile("Recency", [0.25, 0.5, 0.75], 0)
f_quartile = rfm_table.approxQuantile("Frequency", [0.25, 0.5, 0.75], 0)
m_quartile = rfm_table.approxQuantile("Monetary", [0.25, 0.5, 0.75], 0)


# In[446]:


# calculate Recency based on quartile

rfm_table = rfm_table.withColumn("R_Quartile",                                  when(col("Recency") >= r_quartile[2] , 1).                                 when(col("Recency") >= r_quartile[1] , 2).                                 when(col("Recency") >= r_quartile[0] , 3).                                 otherwise(4))


# In[447]:


# calculate Frequency based on quartile

rfm_table = rfm_table.withColumn("F_Quartile",                                  when(col("Frequency") > f_quartile[2] , 4).                                 when(col("Frequency") > f_quartile[1] , 3).                                 when(col("Frequency") > f_quartile[0] , 2).                                 otherwise(1))


# In[448]:


# calculate Monetary based on quartile

rfm_table = rfm_table.withColumn("M_Quartile",                                  when(col("Monetary") >= m_quartile[2] , 4).                                 when(col("Monetary") >= m_quartile[1] , 3).                                 when(col("Monetary") >= m_quartile[0] , 2).                                 otherwise(1))


# In[449]:


# combine the scores (R_Quartile, F_Quartile,M_Quartile) together.

rfm_table = rfm_table.withColumn("RFM_Score", concat(col("R_Quartile"), col("F_Quartile"), col("M_Quartile")))


# In[450]:


rfm_table.show(5)


# # 4 RFM Analysis

# In[457]:


# Best customers

rfm_table.select("CustomerID").where("RFM_Score == 444").show(11)


# In[467]:


# group by RFM Score

grouped_by_rfmscore = rfm_table.groupBy("R_Quartile", "F_Quartile", "M_Quartile").count().orderBy("count", ascending=False)


# In[468]:


# convert Spark dataframe to pandas in order to visualize data

grouped_by_rfmscore_pandas = grouped_by_rfmscore.toPandas()


# In[469]:


grouped_by_rfmscore_pandas


# In[453]:


grouped_by_rfmscore_pandas['RFM_Score'] = "Seg " + grouped_by_rfmscore_pandas['RFM_Score'].map(str)


# In[462]:


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

