#!/bin/bash

# Validate prestatus
is_logstash_running() {
    local PID=$(pgrep -f "kafka-logstash")
    if [ -n "$PID" ]; then
        return 0
    else
        return 1
    fi
}

# Start Logstash app
start_logstash_job() {
    if is_logstash_running; then
        echo "Logstash kpi job is already running."
    else
        /opt/logstash/bin/logstash -f /opt/logstash/data/kafka-logstash.conf > /opt/logstash/data/logstash_error.log &
        echo "Logstash kpi job started."
    fi
}

# Stop Logstash app
stop_logstash_job() {
    local PID=$(pgrep -f "kafka-logstash")
    if [ -n "$PID" ]; then
        kill "$PID"
        echo "Logstash kpi job stopped."
    else
        echo "Logstash kpi job is not running."
    fi
}

# Status Logstash app
check_logstash_job_status() {
    if is_logstash_running; then
        local PID=$(pgrep -f "kafka-logstash")
        echo "Logstash kpi job is running with PID: $PID."
    else
        echo "Logstash kpi job is not running."
    fi
}

# Trigger argument (start - stop - status)
if [ "$1" = "start" ]; then
    start_logstash_job
elif [ "$1" = "stop" ]; then
    stop_logstash_job
elif [ "$1" = "status" ]; then
    check_logstash_job_status
else
    echo "Usage: $0 [start|stop|status]"
    exit 1
fi