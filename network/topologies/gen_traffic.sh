#!/bin/bash
###
 # @Description: 
 # @Version: 1.0
 # @Autor: Ian Yang
 # @Date: 2024-04-26 00:50:21
 # @LastEditors: Ian Yang
 # @LastEditTime: 2024-04-26 15:34:56
### 

# If not enough arguments supplied
if [ $# -lt 2 ]; then
    echo "usage bash traffic.sh <target_ip> <flood_type>"
else 
    TARGET_IP="$1"
    FLOOD_TYPE="$2"
    FILEDR="$3"
    PIDS=$(ps aux | grep "mininet:h" | cut -c5-16 | tail -n+2)
    N_PIDS=$(echo $PIDS | wc -w)

    mkdir -p /var/run/netns

    trap "exit" INT TERM ERR
    trap "rm -rf /var/run/netns && kill 0" EXIT

    I=1
    while read PID; do
        ln -sf /proc/$PID/ns/net /var/run/netns/$PID
        ip netns exec $PID bash $FILEDR $TARGET_IP $FLOOD_TYPE &
        
    done <<< "$PIDS"
    wait
fi
