# NTU SDN 2024

## DDoS attacks detection and mitigation in a Software-Defined Network.

Starting from the build of the network topology using Mininet, I make use of the the Ryu ofctl rest API (https://ryu.readthedocs.io/en/latest/app/ofctl_rest.html) 
to aggregate flows from Open vSwitch switches, delete them and add new ones.



### Training

The parameters used to train are:

- Speed of source IPs;
- Standard deviation of the number of packets per flow entry;
- Standard deviation of the number of bytes per flow entry;
- Speed of creation of flow entries;
- Ratio of pair-flow.

To simulate normal traffic the script ```gen_traffic.sh``` sets Mininet hosts network namespace visible and runs a script to each one that makes use of ```hping3``` to send a random number of icmp packets with different bytes.
On the other hand, the attack traffic uses 
```
hping3 --flood --rand-source --icmp h_target
```
from a Mininet host.

Then run 
```
python collect_data.py $Target_IP
```
on another terminal to sample the flow to target ip.

In order to calulate features, in the Open vSwitch are installed by C1 flow entries with src IP, dst IP as matching field. Then, from each flow entry it's possible 
to sample the packet count and byte count using the Ryu ofctl API. The python script ```collect_data.py``` samples the switch every 3s through the API and builds the dataset.

After that, you can get `dataset.json`.

Use `dataset.json` to train your model through below command
```
cd training/classifier
python train.py
```


### Application

The python-based application uses the MVC pattern to better organize the code and its main class is ```DDoSControllerThread```.
This class at runtime is a unique thread and it's implemented by an Asynchronous Final State Machine.
The states are 3:
- UNCERTAIN;
- NORMAL: in this state the classifier predicted the traffic as normal. The features are real-time plotted in a Tkinter window and legitimate source IPs are stored in an array;
- ANOMAOUS: the classifier predicted the traffic as attack. The features are plotted and mitigation is applied. The mitigation consists of the install of a DROP entry that matches with all packets with dst IP equal to target host IP. Then, the connection for all the legitimate IPs stored in the NORMAL state is maintained to allow them to keep the connection with the server.


### Experimentation topology



Start the controllers in the host:

C1: 
``` 
ryu-manager --ofp-tcp-listen-port 6653 c1.py
```

C2: 
```
ryu-manager --ofp-tcp-listen-port 6633 c2.py ofctl_rest.py
```

Start the topology in the guest in order to maintain CPU resources when the attack occurs and run normal traffic script:

```
sudo mn --custom mn_ddos_topology.py --switch ovsk \
  --controller=remote,ip=192.168.1.5:6653 \
  --controller=remote,ip=192.168.1.5:6633 --topo ddostopo
```
```
sudo bash gen_traffic.sh $Target_IP
```


In conclusion, start the application.


### use in new machine
ubuntu 20.04<br>
using ryu controller 
install ryu in 2024
```
apt install gcc python-dev libffi-dev libssl-dev libxml2-dev libxslt1-dev zlib1g-dev
```
```
python3 -m pip install ryu
```
```
sudo apt install ryu-bin
```
```
pip3 uninstall eventlet
```
```
pip3 install eventlet==0.30.2
```
check it work
```
sudo ryu-manager ryu.app.simple_switch_13
```
if success, you will see like this<br>
![image](https://github.com/ianyang66/NTU-SDN-2024/assets/106331489/e8fd6d62-ab2f-49b2-ad39-a0f5e70ddfc1)

install mininet
```
git clone https://github.com/mininet/mininet.git
```
```
cd mininet/util/
```
```
sudo ./install.sh
```
install done<br>
using on localhost<br>
each controller need to execute in different terminal at controller file using the code above<br>
open another terminal to execute mininet 
```
sudo mn --custom mn_ddos_topology.py --topo ddostopo --switch ovsk  --controller=remote,ip=127.0.0.1:6653  --controller=remote,ip=127.0.0.1:6633 
```
**remeneber**  mininet needs to be run after two controller. if not mininet will not find the controller<br>
befor runnning the program you need to install hping3
```
sudo apt-get install hping3
```
i am confuse how to use sudo bash gen_traffic.sh<br>
because of using in localhost, the ipv4 address of h2 is 10.0.0.2 <br>
and need to add the method "UDP".<br>
so the total command will like this
```
sudo bash gen_traffic.sh 10.0.0.2 "UDP"
```
then i get the response like "Cannot open network namespace "3077": No such file or directory"<br>
what is that mean <br>
let's put it aside <br>
because it's still running <br>
then i get
![image](https://github.com/ianyang66/NTU-SDN-2024/assets/106331489/c025dba9-a18b-4fc7-8cf8-316c63c0196e)
what happened
