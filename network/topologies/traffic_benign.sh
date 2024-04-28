#!/bin/bash
###
 # @Description: 
 # @Version: 1.0
 # @Autor: Ian Yang
 # @Date: 2024-04-26 14:24:38
 # @LastEditors: Ian Yang
 # @LastEditTime: 2024-04-26 13:46:22
### 

# If no argument supplied
if [ $# -lt 1 ]
  then
    echo "usage bash benign_traffic.sh <target_ip>"
else 
    TARGET_IP="$1"
    while true; do
        N_PACKETS=$(shuf -i 10-20 -n 1)
        PAUSE=$(shuf -i 1-10 -n 1)
        ping -c $N_PACKETS "$TARGET_IP"
        iperf -p 5050 -c "$TARGET_IP"
        iperf -p 5051 -u -c "$TARGET_IP"
        sleep $PAUSE
    done
fi