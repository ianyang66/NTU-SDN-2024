#!/bin/bash
###
 # @Description: 
 # @Version: 1.0
 # @Autor: Ian Yang
 # @Date: 2024-04-26 00:47:31
 # @LastEditors: Ian Yang
 # @LastEditTime: 2024-04-26 00:23:47
### 

# If no argument supplied
if [ $# -lt 2 ]
  then
    echo "usage bash traffic.sh <target_ip> <flood_type>"
else 
    TARGET_IP="$1"
    FLOOD_TYPE="$2"
    while true; do
        N_PACKETS=$(shuf -i 10-20 -n 1)
        N_BYTES=$(shuf -i 150-200 -n 1)
        PAUSE=$(shuf -i 1-10 -n 1)
        case $FLOOD_TYPE in
            "UDP")
                sudo hping3 -c $N_PACKETS -d $N_BYTES --udp --fast "$TARGET_IP"
                ;;
            "ICMP")
                sudo hping3 -c $N_PACKETS -d $N_BYTES --icmp --fast "$TARGET_IP"
                ;;
            "TCP-SYN")
                sudo hping3 -c $N_PACKETS -d $N_BYTES --syn --fast "$TARGET_IP"
                ;;
            *)
                echo "Invalid flood type. Please choose UDP, ICMP, or TCP-SYN."
                exit 1
                ;;
        esac
        sleep $PAUSE
    done
fi