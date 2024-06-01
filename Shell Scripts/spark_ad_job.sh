#!/bin/bash

# Validate prestatus
is_spark_job_running() {
    local PID=$(pgrep -f "SparkSubmit.*anomaly_detection")
    if [ -n "$PID" ]; then
        return 0
    else
        return 1
    fi
}

# Start Spark Job
start_spark_job() {
    if is_spark_job_running; then
        echo "Spark anomaly detection job is already running."
    else
        nohup spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 --conf spark.driver.extraJavaOptions="-Dlog4j.configuration=file:/opt/bitnami/spark/tfm/anomaly_det/log4j-ad.properties" /opt/bitnami/spark/tfm/anomaly_det/anomaly_detection-v2.py > /opt/bitnami/spark/tfm/anomaly_det/spark_ad_log.log &
		echo "Spark anomaly detection job started."
    fi
}


# Stop Spark Job
stop_spark_job() {
    PID=$(pgrep -f "SparkSubmit.*anomaly_detection")
    if [ -n "$PID" ]; then
        kill "$PID"
        echo "Spark anomaly detection job stopped."
    else
        echo "Spark anomaly detection job is not running."
    fi
}

# Status Spark Job
check_spark_job_status() {
    PID=$(pgrep -f "SparkSubmit.*anomaly_detection")
    if [ -n "$PID" ]; then
        echo "Spark anomaly detection job is running with PID: $PID."
    else
        echo "Spark anomaly detection job is not running."
    fi
}

# Trigger argument (start - stop - status)
if [ "$1" = "start" ]; then
    start_spark_job
elif [ "$1" = "stop" ]; then
    stop_spark_job
elif [ "$1" = "status" ]; then
    check_spark_job_status
else
    echo "Usage: $0 [start|stop|status]"
    exit 1
fi
