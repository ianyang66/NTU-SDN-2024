#!/bin/bash
###
 # @Description: 
 # @Version: 1.0
 # @Autor: Ian Yang
 # @Date: 2024-04-26 00:45:51
 # @LastEditors: Ian Yang
 # @LastEditTime: 2024-04-26 00:21:58
### 
# If no argument supplied
if [ $# -eq 0 ]; then
    echo "usage: sudo bash traffic.sh <target_ip> <flood_type>"
else
    TARGET_IP="$1"
    FLOOD_TYPE="$2"
    while true; do
        N_PACKETS=$(shuf -i 10-20 -n 1)
        N_BYTES=$(shuf -i 150-200 -n 1)
        PAUSE=$(shuf -i 1-10 -n 1)
        case $FLOOD_TYPE in
            "UDP")
                hping3 --udp -c $N_PACKETS -d $N_BYTES "$TARGET_IP"
                ;;
            "ICMP")
                hping3 --icmp -c $N_PACKETS -d $N_BYTES "$TARGET_IP"
                ;;
            "TCP-SYN")
                hping3 --syn -c $N_PACKETS -d $N_BYTES "$TARGET_IP"
                ;;
            *)
                echo "Invalid flood type. Please choose UDP, ICMP, or TCP-SYN."
                exit 1
                ;;
        esac
        sleep $PAUSE
    done
fi