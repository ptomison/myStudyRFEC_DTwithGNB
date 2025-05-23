# -*- coding: utf-8 -*-
"""
Created on Fri May  2 09:46:34 2025

@author: pauli
"""

import math
from mininet.topo import Topo


class FatTree(Topo):
    """Three-layer fat-tree with 2 pods"""

    def build(self, k=2):
        # Add nodes
        ss = {i: self.addSwitch(f's{i}') for i in range(1, 5*k+1)}
        hs = {i: self.addHost(f'h{i}') for i in range(1, 4*k+1)}

        # Add links of Edge layer
        for i in range(1, 4*k+1):
            j = int(math.ceil(i/2))
            self.addLink(hs[i], ss[j])

        # Add links of Aggregation layer
        for i in range(1, 2*k+1):
            j = i+4
            l = j+1 if i % 2 == 1 else j-1
            self.addLink(ss[i], ss[j])
            self.addLink(ss[i], ss[l])

        # Add links of Core layer
        for i in range(2*k+1, 4*k+1):
            self.addLink(ss[i], ss[5*k-1])
            self.addLink(ss[i], ss[5*k])
            
class FatTreeTopo(Topo):

    # build a fat tree topo of size k
    def __init__(self, k):
        super(FatTreeTopo, self).__init__()

        self.k = k

        pods = [self.make_pod(i) for i in range(k)]

        for core_num in range((k/2)**2):
            dpid = location_to_dpid(core=core_num)
            s = self.addSwitch('c_s%d'%core_num, dpid=dpid)

            stride_num = core_num // (k/2)
            for i in range(k):
                self.addLink(s, pods[i][stride_num])

    
    # makes a single pod with its k switches and (k/2)^2 hosts
    def make_pod(self, pod_num):
        lower_layer_switches = [
            self.addSwitch('p%d_s%d'%(pod_num, i), dpid=location_to_dpid(pod=pod_num, switch=i))
            for i in range(self.k / 2)
        ]

        for i, switch in enumerate(lower_layer_switches):
            for j in range(2, self.k / 2 + 2):
                h = self.addHost('p%d_s%d_h%d'%(pod_num, i, j),
                    ip='10.%d.%d.%d'%(pod_num, i, j),
                    mac=location_to_mac(pod_num, i, j))
                self.addLink(switch, h)
        
        upper_layer_switches = [
            self.addSwitch('p%d_s%d'%(pod_num, i), dpid=location_to_dpid(pod=pod_num, switch=i))
            for i in range(self.k / 2, self.k)
        ]

        for lower in lower_layer_switches:
            for upper in upper_layer_switches:
                self.addLink(lower, upper)

        return upper_layer_switches


topas = {
        'fattree' : FatTreeTopo,
}

