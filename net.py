import sys
import os
import csv
from subprocess import Popen
from time import sleep, time
from mininet.log import setLogLevel, info
from mininet.net import Mininet
from mininet.util import dumpNodeConnections
from mininet.cli import CLI
#from gui import gui
from topo import FatTree
from topo import FatTreeTopo
from mininet.log import setLogLevel, info


class Net:
    def __init__(self, opt1='--flood', opt2='--udp', itfs=[]):
        self.idle_dur = 5
        self.attack_dur = 5
        self.tmp = 'tmp.txt'
        self.opt1 = opt1
        self.opt2 = opt2
        self.opt3 = '-P'
        self.opt4 = '--interval'
        self.opt5 = 'u100'
        self.opt6 = '--count'
        self.opt7 = '10'
        self.itfs = itfs
        self.ack = '-A'
        self.mode = '-0'
        self.udpMode = '-2'
        self.scanMode = '-8'
        self.listenMode = '-9'
        self.data = {}

    def run(self):
        """Start simulation"""
        self.remove_tmp()
        self.clean_net()
        self.start_net()
        self.start_monitor()
        self.aet = time()
        for i in range(1, 11):
            print(f"Slow Push Attack ", i)
            self.start_slow_attack()
            sleep(self.attack_dur)
        for i in range(1, 11):
            print(f"Slow ACK Attack ", i)
            self.start_attack()
            sleep(self.attack_dur)
        sleep(self.idle_dur)
        self.aet = time()
        self.stop_monitor()
        self.fill_data()
        self.remove_tmp()
        self.stop_net()
        #self.plot()

    def clean_net(self):
        """Clean mininet to allow to create new topology"""
        info('*** Clean net\n')
        cmd = ("mn -c")
        Popen(cmd, shell=True).wait()

    def start_net(self):
        """Build the topology and initialize the network"""
        info('*** Building the network topology\n')
        self.net = Mininet(FatTree())
        self.net.start()
        for i in range(1, 11):
            s = self.net.get(f's{i}')
            s.cmd(f'ovs-vsctl set bridge s{i} stp-enable=true')
        print("Dumping host connections")
        dumpNodeConnections(self.net.hosts)
        print("Testing network connectivity")
        self.net.pingAll()
        self.net.pingAll()

    def stop_net(self):
        """Stop mininet with current network"""
        self.net.stop()

    def start_monitor(self):
        """Run tcpdump bwm-ng in background to measure network load and write to a file"""
        info('*** Start monitor\n')
        cmd = f"sudo tcpdump -i lo -w tcpdump.pcap > {self.tmp} &"
        Popen(cmd, shell=True).wait()
        cmd = f"bwm-ng -o csv -T rate -C ',' > {self.tmp} &"
        Popen(cmd, shell=True).wait()
      

    def stop_monitor(self):
        """Kill all running instances of tcpdump and bwm-ng"""
        info('*** Stop monitor\n')
        cmd = "killall tcpdump"
        Popen(cmd, shell=True).wait()
        cmd = "killall bwm-ng"
        Popen(cmd, shell=True).wait()

    def start_attack(self):
        """Attack from h1,h5 to h2,h3,h7,h8 by running instances of hping3 in background"""
        info('*** Start attack\n')
        h1 = self.net.get('h1')
        h5 = self.net.get('h5')
        ip2 = self.net.get('h2').IP()
        ip3 = self.net.get('h3').IP()
        ip7 = self.net.get('h7').IP()
        ip8 = self.net.get('h8').IP()
        h1.cmd(f"hping3 {self.ack} {self.opt4} {self.opt5} {self.opt6} {self.opt7} {ip2} &")
        h1.cmd(f"hping3 {self.ack} {self.opt4} {self.opt5} {self.opt6} {self.opt7} {ip3} &")
        h5.cmd(f"hping3 {self.ack} {self.opt4} {self.opt5} {self.opt6} {self.opt7} {ip7} &")
        h5.cmd(f"hping3 {self.ack} {self.opt4} {self.opt5} {self.opt6} {self.opt7} {ip8} &")

    def start_slow_attack(self):
        """Slow Attack from h2,h4 to h7,h8 by running instances of hping3 in intervals in the backgroud"""
        info('*** Start Slow Push Attack***\n')
        h2 = self.net.get('h2')
        h4 = self.net.get('h4')
        ip7 = self.net.get('h7').IP()
        ip8 = self.net.get('h8').IP()
        info('***h2 Targeting h7***\n')
        h2.cmd(f"hping3 {self.opt3} {self.opt4} {self.opt5} {self.opt6} {self.opt7} {ip7} &")
        #print(f"the slow attack is {self.opt3} {self.opt4} {self.opt5} {self.opt6} {self.opt7}")
        info('***h2 Targeting h8***\n')
        h2.cmd(f"hping3 {self.opt3} {self.opt4} {self.opt5} {self.opt6} {self.opt7} {ip8} &")
        info('***h4 Targeting h7***\n')
        h4.cmd(f"hping3 {self.opt3} {self.opt4} {self.opt5} {self.opt6} {self.opt7}{ip7} &")
        info('***h4 Targeting h8***\n')
        h4.cmd(f"hping3 {self.opt3} {self.opt4} {self.opt5} {self.opt6} {self.opt7} {ip8} &")

    def stop_attack(self):
        """Kill all running instances of hping3"""
        info('*** Stop attack\n')
        cmd = "killall hping3"
        Popen(cmd, shell=True).wait()

    def fill_data(self):
        """Read the output of bwm-ng from a file"""
        info('***outputting the bwm-ng into a csv file\n')
        with open(self.tmp) as csvf:
            csvr = csv.reader(csvf, delimiter=',')
            for row in csvr:
                key = row[1]
                tme = float(row[0])
                load = float(row[4]) * 8
                if key in self.data:
                    self.data[key]['time'].append(tme)
                    self.data[key]['load'].append(load)
                else:
                    self.data[key] = {}
                    self.data[key]['time'] = []
                    self.data[key]['load'] = []           

    def plot(self):
        """Pass the loaded output of bwm-ng to gui to plot"""
        info('*** Plot\n')
        self.itfs = [t for t in self.itfs if t in self.data]
        gui(self.data, (self.ast, self.aet), self.itfs)

    def remove_tmp(self):
        """Remove the output file of bwm-ng if already exists"""
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def stop_all(self):
        """Kill all running instances of hping3, bwm-ng and stop the net"""
        try:
            self.stop_attack()
            self.stop_monitor()
            self.remove_tmp()
            self.stop_net()
        except Exception as e:
            pass


def main():
    """Run the script"""
    setLogLevel('info')
    opt1 = sys.argv[1] if len(sys.argv) > 1 else '--flood'
    opt2 = sys.argv[2] if len(sys.argv) > 2 else '--udp'
    itfs = sys.argv[8:] if len(sys.argv) > 3 else []
    n = Net(opt1, opt2, itfs)
    try:
        n.run()
    except KeyboardInterrupt:
        n.stop_all()


if __name__ == '__main__':
    main()
