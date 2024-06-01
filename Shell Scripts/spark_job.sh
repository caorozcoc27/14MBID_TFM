#!/bin/bash

# Validate prestatus
is_spark_job_running() {
    local PID=$(pgrep -f "SparkSubmit.*kafka-ops")
    if [ -n "$PID" ]; then
        return 0
    else
        return 1
    fi
}

# Start Spark Job
start_spark_job() {
    if is_spark_job_running; then
        echo "Spark kpi calculation job is already running."
    else
        nohup spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 --conf spark.driver.extraJavaOptions="-Dlog4j.configuration=file:/opt/bitnami/spark/tfm/log4j.properties" /opt/bitnami/spark/tfm/kafka-ops-v17.py > /opt/bitnami/spark/tfm/spark_log.log &
        echo "Spark kpi calculation job started."
    fi
}


# Stop Spark Job
stop_spark_job() {
    PID=$(pgrep -f "SparkSubmit.*kafka-ops")
    if [ -n "$PID" ]; then
        kill "$PID"
        echo "Spark kpi calculation job stopped."
    else
        echo "Spark kpi calculation job is not running."
    fi
}

# Status Spark Job
check_spark_job_status() {
    PID=$(pgrep -f "SparkSubmit.*kafka-ops")
    if [ -n "$PID" ]; then
        echo "Spark kpi calculation job is running with PID: $PID."
    else
        echo "Spark kpi calculation job is not running."
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
