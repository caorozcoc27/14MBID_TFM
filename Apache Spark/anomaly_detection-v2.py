
#!/usr/bin/python3

# Carga de librerias
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pickle
import sklearn
from sklearn.ensemble import IsolationForest
from pyspark.ml import Pipeline
import joblib
import numpy as np


# Carga del modelo
model_if = joblib.load("/opt/bitnami/spark/tfm/anomaly_det/model_bcn_anomaly_detection.pkl")

# Funciones de predicción
def predict_anomaly(kpiValues):
    prediction = model_if.predict(np.array(kpiValues).reshape(-1, 1))
    return prediction

predict_anomaly_udf = udf(lambda x: predict_anomaly(x).tolist(), ArrayType(IntegerType()))

def process_df_pred(df):
    prediction_df = df.withColumn("prediction", predict_anomaly_udf(df["kpiValue"]))
    return prediction_df

# Configuración de Kafka
kafka_bootstrap_servers = "kafka-cluster-kafka-brokers.kafka.svc.cluster.local:9092"
input_kafka_topic = "metrics"
output_kafka_topic = "anomalies"
checkpoint_location = "/opt/bitnami/spark/tfm/anomaly_det/checkpoints"

# Configuración de la sesión de Spark
spark = SparkSession \
    .builder \
    .appName("anomalyDetectionJob") \
    .getOrCreate()

# Lectura de datos de Kafka
input_kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", input_kafka_topic) \
    .option("startingOffsets", "latest") \
    .load()

# Ajuste de los mensajes de Kafka
decoded_kafka_df = input_kafka_df \
    .selectExpr("CAST(value AS STRING) as json_value") \
    .selectExpr("from_json(json_value, 'neId STRING, neType STRING, kpiName STRING, kpiValue FLOAT, vendorName STRING, granularity INT, startTime LONG, category STRING, unit  STRING, neName  STRING, neManufacturer  STRING, neModel  STRING, operativeState  STRING, comunityName  STRING, provinceName  STRING, cityName  STRING, postalCode  STRING, locationId STRING, latitude FLOAT, longitude FLOAT') as data") \
    .select("data.startTime", "data.neId", "data.neType", "data.vendorName", "data.kpiName", "data.kpiValue", "data.category", "data.neName", "data.comunityName", "data.provinceName", "data.cityName", "data.locationId") \
    .withColumn("startTime", expr("to_timestamp(from_unixtime(startTime / 1000))"))
 
# Filtrado de datos por neId y kpiName
filtered_kafka_df = decoded_kafka_df.filter(decoded_kafka_df.neId == "BCN014_01").filter(decoded_kafka_df.kpiName == "DLPDCP_VOLUME")

# Predicción de anomalías
predicted_kafka_df = process_df_pred(filtered_kafka_df)

# Validación de anomalías
anomaly_kafka_df = predicted_kafka_df.withColumn("anomaly", when(col("prediction").getItem(0) == -1, True).otherwise(False))

# Ajuste formato de salida
transformed_df = anomaly_kafka_df \
    .selectExpr("startTime", "neId", "neType", "vendorName", "kpiName", "cast(kpiValue as decimal(38,3)) as kpiValue", "category", "neName", "comunityName", "provinceName", "cityName", "locationId", "anomaly") \
    .withColumn("startTime", unix_timestamp("startTime") * 1000) \
    .selectExpr("to_json(struct(*)) as value")

# Publicación de anomalías en Kafka
query = transformed_df \
    .writeStream \
    .format("kafka") \
    .outputMode("update") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("topic", output_kafka_topic) \
    .option("checkpointLocation", checkpoint_location) \
    .trigger(processingTime="30 seconds") \
    .start()

query.awaitTermination()
