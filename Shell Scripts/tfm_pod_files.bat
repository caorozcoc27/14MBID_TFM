
REM ---- TFM files ----

REM -- NiFi
kubectl cp ./counter_files.zip nifi/nifi-0:/opt/nifi/data/counter_files.zip --container server

REM -- Spark
kubectl exec -it spark-master-0 -n spark -- mkdir -p /opt/bitnami/spark/tfm/checkpoints
kubectl exec -it spark-master-0 -n spark -- mkdir -p /opt/bitnami/spark/tfm/anomaly_det/checkpoints
kubectl exec -it spark-master-0 -n spark -- pip install kafka-python
kubectl exec -it spark-master-0 -n spark -- pip install joblib
kubectl exec -it spark-master-0 -n spark -- pip install scikit-learn==1.4.2
kubectl exec -it spark-master-0 -n spark -- pip install numpy==1.26.4
kubectl cp ./kafka-ops-v17.py spark/spark-master-0:/opt/bitnami/spark/tfm/kafka-ops-v17.py
kubectl cp ./postgresql-42.7.0.jar spark/spark-master-0:/opt/bitnami/spark/tfm/postgresql-42.7.0.jar
kubectl cp ./postgresql-42.7.0.jar spark/spark-master-0:/opt/bitnami/spark/jars/postgresql-42.7.0.jar
kubectl cp ./anomaly_detection-v2.py spark/spark-master-0:/opt/bitnami/spark/tfm/anomaly_det/anomaly_detection-v2.py
kubectl cp ./model_bcn_anomaly_detection.pkl spark/spark-master-0:/opt/bitnami/spark/tfm/anomaly_det/model_bcn_anomaly_detection.pkl
kubectl cp ./log4j.properties spark/spark-master-0:/opt/bitnami/spark/tfm/log4j.properties
kubectl cp ./log4j-ad.properties spark/spark-master-0:/opt/bitnami/spark/tfm/anomaly_det/log4j-ad.properties
kubectl cp ./spark_job.sh spark/spark-master-0:/opt/bitnami/spark/tfm/spark_job.sh
kubectl exec -it spark-master-0 -n spark -- chmod +x /opt/bitnami/spark/tfm/spark_job.sh
kubectl cp ./spark_ad_job.sh spark/spark-master-0:/opt/bitnami/spark/tfm/anomaly_det/spark_ad_job.sh
kubectl exec -it spark-master-0 -n spark -- chmod +x /opt/bitnami/spark/tfm/anomaly_det/spark_ad_job.sh


REM -- Logstash
kubectl cp ./kafka-logstash.conf elk/logstash-logstash-0:/opt/logstash/data/kafka-logstash.conf
kubectl cp ./anomaly-logstash.conf elk/logstash-logstash-0:/opt/logstash/data/anomaly-logstash.conf
kubectl cp ./logstash.yml elk/logstash-logstash-0:/opt/logstash/data/logstash.yml
kubectl cp ./logstash.yml elk/logstash-logstash-0:/opt/logstash/config/logstash.yml
kubectl cp ./logstash_job.sh elk/logstash-logstash-0:/opt/logstash/data/logstash_job.sh
kubectl cp ./logstash_ad_job.sh elk/logstash-logstash-0:/opt/logstash/data/logstash_ad_job.sh
kubectl exec -it logstash-logstash-0 -n elk -- mkdir -p /opt/logstash/data/exec
kubectl exec -it logstash-logstash-0 -n elk -- mkdir -p /opt/logstash/data/exec_ad
kubectl exec -it logstash-logstash-0 -n elk -- chmod +x /opt/logstash/data/logstash_job.sh
kubectl exec -it logstash-logstash-0 -n elk -- chmod +x /opt/logstash/data/logstash_ad_job.sh


pause