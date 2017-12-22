
# Chuanyue You
# HFT section volatility and realize volatility

from datetime import datetime,timedelta
from pyspark.sql import functions as f
from pyspark.sql.types import StringType
from pyspark.sql.functions import col,lag,log1p,sqrt
from pyspark.sql.window import Window

sc = SparkContext.getOrCreate()

data = spark.read.csv('HFT_baby_data.csv', header=True)

data = data.select(data._c0.alias('INDEX'),data.SYMBOL,data.DATE,data.TIME,data.PRICE.cast('float'),data.SIZE.cast('float'))

def generate_end_points(start, end, interval):
    """
    function to generate ending points of every time interval between start time and end time
    start: a string represents starting time point, use 24-hour-clock, 'HOUR:MIN:SEC' e.g. '9:30:00'
    end: a string represents ending time point, use 24-hour-clock, 'HOUR:MIN:SEC' e.g. '16:00:00'
    interval: int, time interval
    """
    start = datetime.strptime(start, '%H:%M:%S')
    end = datetime.strptime(end, '%H:%M:%S')
    # list to store end point of each time interval
    endPoints = list()
    num_intervals = (end-start).total_seconds()/60/interval
    # initialize first end point as opean time
    end_point=start
    for i in range(int(num_intervals)):
        end_point = end_point + timedelta(minutes=interval)
        endPoints.append(end_point)
        
    return endPoints

# market normal open time
start = '9:30:00'
# market normal close time
end = '16:00:00'
# 30 mins time interval
interval = 30
# generate end points
endPoints = generate_end_points(start, end, interval)

def assign_interval(timestamp):
    """
    function to assign every timestamp in data to the time interval it belongs to
    a timestamp belongs to a specific time interval means:
    start_time of this time interval <= timestamp < start_time of this time interval
    """
    timestamp = datetime.strptime(timestamp, '%H:%M:%S')
    for i in range(len(endPoints)):
        if timestamp < endPoints[i]:
            break
    return i

from pyspark.sql import functions as f
from pyspark.sql.types import StringType

# convert into user defined function so that can be applied to pysaprk dataframe
udf_assign_interval = f.udf(assign_interval,StringType())

data = data.withColumn('INTERVAL',udf_assign_interval('TIME'))

# calculate mean price of each time intervel according to symbol, date, interval
data = data.groupby(['SYMBOL','DATE','INTERVAL']).agg({'PRICE':'mean'})
data = data.withColumnRenamed('avg(PRICE)', 'AVG_PRICE')

data = data.select(data.SYMBOL,data.DATE,data.INTERVAL.cast('double'),data.AVG_PRICE)

data = data.orderBy(["SYMBOL","DATE","INTERVAL"], ascending=[1, 1])

# apply window function to get previous time interval avg_price
w = Window().partitionBy(col('SYMBOL')).orderBy([col('SYMBOL'),col('DATE'),col('INTERVAL')])
data = data.select("*", lag('AVG_PRICE').over(w).alias('PRE_AVG_PRICE'))

# compute log return
data = data.withColumn('U_SEQUENCE',log1p(data.AVG_PRICE/data.PRE_AVG_PRICE-1.))

data = data.withColumn('SQUARE_U_SEQUENCE', data.U_SEQUENCE * data.U_SEQUENCE)

window = Window().partitionBy("SYMBOL").rowsBetween(-(len(endPoints)-1), 0).orderBy([col('SYMBOL'),col('DATE'),col('INTERVAL')])
new_data = data.select(data.SYMBOL,data.DATE,data.INTERVAL,data.U_SEQUENCE,data.SQUARE_U_SEQUENCE,f.sum('U_SEQUENCE').over(window).alias('SUM_U'),f.sum('SQUARE_U_SEQUENCE').over(window).alias('SQUARE_U_SUM'))

N = float(len(endPoints))
# compute section volatility
new_data = new_data.withColumn("SECTION_VOLATILITY",sqrt(col('SQUARE_U_SUM')/(N-1.) - col('SUM_U')**2/(N*(N-1.))))
SRData = new_data.select(new_data.SYMBOL,new_data.DATE,new_data.SECTION_VOLATILITY,new_data.SQUARE_U_SUM)

SRData = SRData.withColumnRenamed('SQUARE_U_SUM', 'REALIZE_VOLATILITY')

meanSR = SRData.groupby(['SYMBOL','DATE']).agg({'SECTION_VOLATILITY':'mean','REALIZE_VOLATILITY':'mean'})

