# 14MBID_TFM
Este repositorio contiene el código fuente de la solución propuesta en el desarrollo del TFM "Solución para el procesamiento de KPIs de redes móviles y detección de anomalías".

## Contenido

Los archivos de código se han dividido en directorios correspondientes a las distintas herramientas que componen la solución. La descripción de los ficheros existentes se detalla a continuación: 

- Apache Kafka:
	- kafka-cluster-file.yaml: Fichero de configuración del clúster de Kafka
	- kafka-topic-file.yaml: Fichero de configuración de los tópicos
- Apache NiFi:
	- Counters_Ingestion_20240407.xml: Plantilla del flujo de ingestión
- Apache Spark:
	- anomaly_detection-v2.py: Fichero pyspark del proceso de detección de anomalías
	- kafka-ops-v17.py: Fichero pyspark del proceso de cálculo de KPIs
- Elasticsearch:
	- elastic_template.esql: Plantilla de definición del índice de KPIs
- Kibana:
	- Anomalies-Dashboard_v2.ndjson: Fichero de definición del cuadro de mando de anomalías
	- PM-Dashboard_v5.ndjson: Fichero de definición del cuadro de mando de KPIs
- Logstash:
	- anomaly-logstash.conf: Proceso de colecta de anomalías
	- kafka-logstash.conf: Proceso de colecta de KPIs
	- logstash.yaml: Fichero de configuración de la instancia de logstash
- PostgreSQL: 
	- postgresql_tables.sql: Comandos de creación y población de las tablas
	- postgresql-42.7.0.jar: Controlador de conexión JDBC a postgresql
- Python: 
	- KPI Anomaly detection.ipynb: Cuaderno de Jupyter del desarrollo de la detección de anomalías
	- KPI Anomaly detection.py: Código python del desarrollo de la detección de anomalías
	- model_xxx_anomaly_detection.pkl: Modelos de detección de anomalías entrenados y exportados para cada elemento
- Shell Scripts: 
	- log4j.properties: Fichero de configuración de logs para proceso de cálculo de KPIs de Spark
	- log4j-ad.properties: Fichero de configuración de logs para proceso de detección de anomalías de Spark
	- logstash_ad_job.sh: Fichero de control del proceso de colecta de anomalías de logstash
	- logstash_job.sh: Fichero de control del proceso de colecta de KPIs de logstash
	- spark_ad_job.sh: Fichero de control del proceso de detección de anomalías de Spark
	- spark_job.sh: Fichero de control del proceso de cálculo de KPIs de Spark
	- tfm_pod_files.bat: Fichero de preparación inicial de contenedores 
- Source Data:
	- counter_files.zip: Ficheros CSV totales de contadores fuente de cada periodo
	- counter_files_test.zip: Ficheros CSV del conjunto de pruebas de contadores fuente de cada periodo


## Instalación de las herramientas

La instalación de las herramientas de la solución en el clúster de Kubernetes se realiza mediante los Helm Charts oficiales. Para la instalación se utilizan comandos del cliente kubectl y del cliente helm, estos comandos se presentan a continuación:

- Apache Spark
```
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
helm install spark -n spark bitnami/spark --set global.storageClass=default --set master.resources.requests.cpu=100m --set master.resources.limits.memory=2.5Gi --set worker.resources.requests.cpu=100m --version 8.1.6 --create-namespace
```
- Apache NiFi
```
helm repo add cetic https://cetic.github.io/helm-charts
helm repo update
helm install nifi -n nifi cetic/nifi --set nifi.resources.limits.cpu=0.5 --set nifi.resources.limits.memory=3Gi --set jvmMemory=3g --set persistence.enabled=true --set zookeeper.replicaCount=1 --version 1.2.0 --create-namespace
```
- Apache Kafka
```
helm repo add strimzi https://strimzi.io/charts/
helm install strimzi-cluster-operator strimzi/strimzi-kafka-operator -n kafka --set resources.requests.cpu=150m --version 0.38.0 --create-namespace
kubectl apply -n kafka -f kafka-cluster-file.yml
kubectl apply -n kafka -f kafka-topic-file.yml
```
- Postgresql
```
helm install postgresql -n postgresql bitnami/postgresql --set audit.pgAuditLog="all" --set global.postgresql.auth.postgresPassword=fELI5eANGA --version 13.2.24 --create-namespace
```
- ELK
```
helm repo add elastic https://helm.elastic.co
helm repo update
helm install elasticsearch elastic/elasticsearch --namespace elk --set replicas=3 --set antiAffinity=soft --set secret.password=pcKt4SGMM88owSba --set resources.requests.memory=3.5Gi --set resources.limits.memory=3.5Gi --set resources.requests.cpu=250m --version 8.5.1 --create-namespace
helm install kibana elastic/kibana -n elk --set resources.requests.cpu=250m --version 8.5.1
helm install logstash elastic/logstash -n elk --set persistence.enabled=true --set resources.requests.cpu=250m --set resources.limits.memory=3Gi --version 8.5.1
```

## Preparación y ejecución de la solución

A continuación se describe el procedimiento de preparación y ejecución de la solución propuesta: 

1. Iniciar el clúster AKS en el portal de Azure, conectarse al clúster mediante la az cli u otro método y activar port-forward para la GUI de las aplicaciones NiFi y Kibana
2. Ejecutar el archivo tfm_pod_files.bat para la preparación de los contenedores
```
Start-Process -FilePath tfm_pod_files.bat
```
3. Importar el template de NiFi (Counters_Ingestion_20240407.xml) mediante la GUI y activar los servicios del procesador ConvertRecord. 
4. Crear y poblar las tablas de PostgreSQL con los comandos del archivo postgresql_tables.sql.
5. Iniciar los procesos de Spark
```
kubectl exec spark-master-0 -n spark -- sh /opt/bitnami/spark/tfm/spark_job.sh start
kubectl exec spark-master-0 -n spark -- sh /opt/bitnami/spark/tfm/anomaly_det/spark_ad_job.sh start
```
6. Crear la plantilla del índice de elasticsearch del archivo elastic_template.esql.
7. Crear en Kibana los Data View e importar los dashboard de los archivos PM-Dashboard_v4.ndjson y Anomnalies-Dashboard_v1.ndjson.
8. Iniciar los procesos de Logstash
```
kubectl exec logstash-logstash-0 -n elk -- sh /opt/logstash/data/logstash_job.sh start
kubectl exec logstash-logstash-0 -n elk -- sh /opt/logstash/data/logstash_ad_job.sh start
```
9. Iniciar los procesadores del flujo de NiFi. Ajustar el procesador ControlRate según la tasa de procesamiento de datos requerida.
10. Monitorear las distintas herramientas en el clúster, el consumo de recursos, el estado de los procesos y los dashboard de Kibana para validar el correcto funcionamiento.
```
kubectl top nodes
kubectl top pods --all-namespaces
kubectl exec spark-master-0 -n spark -- sh /opt/bitnami/spark/tfm/spark_job.sh status
kubectl exec spark-master-0 -n spark -- sh /opt/bitnami/spark/tfm/anomaly_det/spark_ad_job.sh status
kubectl exec logstash-logstash-0 -n elk -- sh /opt/logstash/data/logstash_job.sh status
kubectl exec logstash-logstash-0 -n elk -- sh /opt/logstash/data/logstash_ad_job.sh status
```



Para detener la solución se sigue el siguiente procedimiento: 

1. Detener los procesadores del flujo de NiFi.
2. Detener los procesos de Spark y de Logstash
```
kubectl exec spark-master-0 -n spark -- sh /opt/bitnami/spark/tfm/spark_job.sh stop
kubectl exec spark-master-0 -n spark -- sh /opt/bitnami/spark/tfm/anomaly_det/spark_ad_job.sh stop
kubectl exec logstash-logstash-0 -n elk -- sh /opt/logstash/data/logstash_job.sh stop
kubectl exec logstash-logstash-0 -n elk -- sh /opt/logstash/data/logstash_ad_job.sh stop
```
3. Detener el clúster AKS en en portal de Azure.






