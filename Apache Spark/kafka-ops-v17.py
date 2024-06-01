
#!/usr/bin/python3

# Carga de librerias
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Configuración de PostgreSQL
driver_path = "/opt/bitnami/spark/tfm/postgresql-42.7.0.jar"
user = "postgres"
pw = "fELI5eANGA" 
postgres_db_url = "jdbc:postgresql://postgresql.postgresql.svc.cluster.local:5432/postgres"
kpi_table = "public.metricslist"
enrichment_table = "public.enrichmentdata"

# Configuración de Kafka
kafka_bootstrap_servers = "kafka-cluster-kafka-brokers.kafka.svc.cluster.local:9092"
input_kafka_topic = "counters"
output_kafka_topic = "metrics"
checkpoint_location = "/opt/bitnami/spark/tfm/checkpoints"

# Configuración la sesión de Spark
spark = SparkSession \
    .builder \
    .appName("SparkPostgresJob") \
    .config("spark.jars", driver_path) \
    .config('spark.driver.extraClassPath', driver_path) \
    .getOrCreate()

spark.sparkContext.addPyFile(driver_path)

# Lectura tabla de KPIs
df_kpis = spark.read \
    .format("jdbc") \
    .option("url", postgres_db_url) \
    .option("dbtable", kpi_table) \
    .option("user", user) \
    .option("password", pw) \
    .option("driver", "org.postgresql.Driver") \
    .load()

# Filtrado de KPIs activos
decoded_kafka_df_kpis = df_kpis.filter(col("enabled") == True)

# Manejo de columnas de contadores
exploded_kpis_df = decoded_kafka_df_kpis.selectExpr("kpiName", "category", "vendor", "unit", "stack(2, 'Num', counterNum, 'Den', counterDen) as (operator, counterCode)")

# Broadcast de datos de KPIs
broadcast_kpis = broadcast(exploded_kpis_df)

# Lectura tabla de enriquecimiento
df_enrichment = spark.read \
    .format("jdbc") \
    .option("url", postgres_db_url) \
    .option("dbtable", enrichment_table) \
    .option("user", user) \
    .option("password", pw) \
    .option("driver", "org.postgresql.Driver") \
    .load()

# Ajuste datos de enriquecimiento
decoded_df_enrichment = df_enrichment.select("neId", "neName", "neManufacturer", "neModel", "operativeState", "comunityName", "provinceName", "cityName", "postalCode", "locationId", "latitude", "longitude")
decoded_df_enrichment = decoded_df_enrichment.withColumnRenamed("neId", "neIdEnr")

# Broadcast de datos de enriquecimiento
broadcast_enrichment = broadcast(decoded_df_enrichment)

# Lectura de contadores desde Kafka
input_kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", input_kafka_topic) \
    .option("startingOffsets", "latest") \
    .load()

# Ajuste de los contadores
decoded_kafka_df = input_kafka_df \
    .selectExpr("CAST(value AS STRING) as json_value") \
    .selectExpr("from_json(json_value, 'neId STRING, neType STRING, counterCode STRING, counterValue LONG, vendorName STRING, granularity INT, startTime LONG') as data") \
    .select("data.*") \
    .withColumn("startTime", expr("to_timestamp(from_unixtime(startTime / 1000))"))

# Cruce con tabla de KPIs y calculo de KPIs
joined_df = decoded_kafka_df.join(broadcast_kpis, (decoded_kafka_df.counterCode == broadcast_kpis.counterCode) & (decoded_kafka_df.vendorName == broadcast_kpis.vendor), "inner") \
    .groupBy(window("startTime", "15 minutes"), "neId", "neType", "vendorName", "granularity", "kpiName", "category", "vendor", "unit") \
    .agg(sum(when(col("operator") == "Num", col("counterValue")).otherwise(0)).alias("numerator_sum"),
         sum(when(col("operator") == "Den", col("counterValue")).otherwise(0)).alias("denominator_sum")) \
    .withColumn("kpiValue", when((col("unit") == "simple"), col("numerator_sum"))
                            .when((col("unit") == "rate") & (col("denominator_sum") != 0), 100 * col("numerator_sum") / col("denominator_sum"))
                            .otherwise(0).cast(DecimalType(38,3)))

# Cruce con inventario de elementos
enriched_df = joined_df.join(broadcast_enrichment, joined_df.neId == broadcast_enrichment.neIdEnr, "left")

# Ajuste formato de salida de KPIs
transformed_df = enriched_df \
    .selectExpr("unix_timestamp(window.start) * 1000 as startTime", "neId", "neType", "vendorName", "granularity", "kpiName", "kpiValue", "category", "unit", "neName", "neManufacturer", "neModel", "operativeState", "comunityName", "provinceName", "cityName", "postalCode", "locationId", "latitude", "longitude") \
    .selectExpr("to_json(struct(*)) as value")

# Publicación de KPIs en Kafka
query = transformed_df \
    .writeStream \
    .format("kafka") \
    .outputMode("update") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("topic", output_kafka_topic) \
    .option("checkpointLocation", checkpoint_location) \
    .trigger(processingTime="60 seconds") \
    .start()

query.awaitTermination()
