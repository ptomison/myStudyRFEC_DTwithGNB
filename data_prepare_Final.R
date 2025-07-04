if(!require(Rcpp)){
	install.packages("Rcpp")
}
if(!require(cli)){
	install.packages("cli")
}
if(!require(rlang)){
	install.packages("rlang")
}
if(!require(vctrs)){
	install.packages("vctrs")
}

if(!require(ipaddress)){
	install.packages("ipaddress")
}

if(!require(tibble)){
	install.packages("tibble")
}

if(!require(stR)){
	install.packages("stR")
}

if(!require(tidyverse)){
	install.packages("tidyverse")
}
if(!require(stringr)){
	install.packages("stringr")  # Install the package
}
if(!require(abind)) {
	install.packages("abind")
}

install.packages("devtools")
install.packages("githubinstall")
install.packages('C:/PhD/DIS9903A/Week 7/iptools-master.zip', lib='C:/Users/pauli/AppData/Local/R/win-library/4.5',repos = NULL)
install.packages("IP", configure.args="--enable-avx2")

library(IP)
library(githubinstall)
library(abind)
library(stringi)
library(stringr)             # Load the package
library(purrr)
library(Rcpp)
library(cli)
library(rlang)
library(vctrs)
library(tibble)
library(ipaddress)
library(tidyr)

directory <- readline(prompt = "Enter your directory name: ")
print(directory)
test <- readline(prompt = "Enter the data file to convert: ")
print(test)
setwd(directory)
getwd()

output <- gsub(".csv", "_convert1.csv", test)
#   read in dataset
test1 <- read.csv(test, header = TRUE, na.strings = "")
head(test1)
row <- nrow(test1)

time = test$Time
address <- test$Source
# special cases in IP source address
# MS-NLB-PhysServer-32_37:88:7a:07
address <- gsub("\\MS-NLB-PhysServer-32_37:88:7a:07", "32:37:88:7a:07", address) 
# PCSSystemtec_4c:e1:2a
address <- gsub("\\PCSSystemtec_4c:e1:2a", "4c:e1:2a", address) 
# PCSSystemtec_b4:51:8f
address <- gsub("\\PCSSystemtec_b4:51:8f", "b4:51:8f", address) 
# PCSSystemtec_30:32:69
address <- gsub("\\PCSSystemtec_30:32:69", "30:32:69", address) 
#PCSSystemtec_be:d6:b5
address <- gsub("\\PCSSystemtec_be:d6:b5", "be:d6:b5", address) 
# PCSSystemtec_76:39:cb
address <- gsub("\\PCSSystemtec_76:39:cb", "76:39:cb", address) 
# PCSSystemtec_80:5c:98
address <- gsub("\\PCSSystemtec_80:5c:98", "80:5c:98", address) 
# PCSSystemtec_d2:95:82
address <- gsub("\\PCSSystemtec_d2:95:82", "d2:95:82", address) 

#MS-NLB-PhysServer-32_7f:cb:c9:96
address <- gsub("\\MS-NLB-PhysServer-32_7f:cb:c9:96", "32:7f:cb:c9:96", address) 
#MS-NLB-PhysServer-16_da:13:20:25
address <- gsub("\\MS-NLB-PhysServer-16_da:13:20:25", "16:da:13:20:25", address) 
#MS-NLB-PhysServer-23_b2:04:ec:4b
address <- gsub("\\MS-NLB-PhysServer-23_b2:04:ec:4b", "23:b2:04:ec:4b", address) 
#MS-NLB-PhysServer-31_a0:85:9b:01
address <- gsub("\\MS-NLB-PhysServer-31_a0:85:9b:01", "31:a0:85:9b:01", address) 
# MS-NLB-PhysServer-32_06:00:a5:47:b4
address <- gsub("\\MS-NLB-PhysServer-32_06:00:a5:47:b4", "32:06:00:a5:47:b4", address) 
# MS-NLB-PhysServer-01_e9:3b:07:28
address <- gsub("\\MS-NLB-PhysServer-01_e9:3b:07:28", "01:e9:3b:07:28", address) 
# MS-NLB-PhysServer-17_c7:00:b2:4e
address <- gsub("\\MS-NLB-PhysServer-17_c7:00:b2:4e", "17:c7:00:b2:4e", address) 

i <- 0
for (i in 1:length(address)) {
   # if address contains letters then chang them to the number
   has_letters <- str_detect(address[i], "[a-zA-Z]")
   temp <- address[i]
   if (has_letters) {
      # convert to ascii represenatative
      address[i] <- chartr("abcdef", "101112131415", temp)
   }
   else
      next
}
address <- gsub("\\:", ".", address) 
Source_address <- gsub("\\.", "", address)

destination <- test$Destination
destination <- gsub("\\Broadcast", "255.255.255.255", destination) 
destination <- gsub("\\Spanning-tree-\\(for\\-bridges\\)\\_00", "10.10.10.10", destination)
destination <- gsub("\\MS-NLB-PhysServer-32_37:88:7a:07", "32:37:88:7a:07", destination)
destination <- replace_na(destination , "999.999.999.999")
i <- 0
for (i in 1:length(destination )) {
   # if address contains letters then chang them to the number
   has_letters <- str_detect(destination[i], "[a-zA-Z]")
   temp <- destination[i]
   if (has_letters) {
      # convert to ascii represenatative
      destination[i] <- chartr("abBcdef", "101112131415", temp)
   }
   else
      next
}
destination <- gsub("\\.", "", destination )
Destination_address <- gsub("\\:", "", destination )

protocol_converted = test$Protocol
protocol_converted <- gsub("\\DNS", 1001, protocol_converted )
protocol_converted <- gsub("\\TCP", 2001, protocol_converted )
protocol_converted <- gsub("\\ARP", 3002, protocol_converted )
protocol_converted <- gsub("\\OpenFlow", 4004, protocol_converted )
protocol_converted <- gsub("\\M1", 5005, protocol_converted )
protocol_converted <- gsub("\\STP", 6006, protocol_converted )
protocol_converted <- gsub("\\DHCP", 7007, protocol_converted )
protocol_converted <- gsub("\\ICMP", 8008, protocol_converted )
protocol_converted <- gsub("\\N1", 9009, protocol_converted )
protocol_converted <- gsub("\\IGMPv3", 1010, protocol_converted )

# Extract the TCP flags from the info data and the Openflow information
# Convert the values to numeric data
info = test$Info
ack <- str_extract(test$Info, "\\[ACK]")
ack_count <- gsub("\\[ACK]", 1, ack)
ack_count <- replace_na(ack_count, "0")
ack_count <- as.numeric(ack_count)

syn <- str_extract(test$Info, "\\[SYN]")
syn_count <- gsub("\\[SYN]", 1, syn)
syn_count <- replace_na(syn_count, "0")
syn_count <- as.numeric(syn_count)

rst_ack <- str_extract(test$Info, "\\[RST, ACK]")
rst_ack_count <- gsub("\\[RST, ACK]", 1, rst_ack)
rst_ack_count <- replace_na(rst_ack_count, "0")
rst_ack_count <- as.numeric(rst_ack_count)

psh <- str_extract(test$Info, "\\[PSH]")
psh_count <- gsub("\\[PSH]", 1, psh)
psh_count <- replace_na(psh_count, "0")
psh_count <- as.numeric(psh_count)

rst <- str_extract(test$Info, "\\[RST]")
rst_count <- gsub("\\[RST]", 1, rst)
rst_count <- replace_na(rst_count, "0")
rst_count <- as.numeric(rst_count)

tcp_dup_ack <- str_extract(test$Info, "\\TCP Dup ACK ") 
tcp_dup_ack_count <- gsub("\\TCP Dup ACK ", 1, tcp_dup_ack)
tcp_dup_ack_count <- replace_na(tcp_dup_ack_count, "0")
tcp_dup_ack_count <- as.numeric(tcp_dup_ack_count)

opt_hello <- str_extract(test$Info, "\\OFPT_HELLO")
opt_hello_count <- gsub("\\OFPT_HELLO", 1, opt_hello)
opt_hello_count <- replace_na(opt_hello_count, "0")
opt_hello_count <- as.numeric(opt_hello_count)

opt_pkt_in <- str_extract(test$Info, "\\OFPT_PACKET_IN")
opt_pkt_in_count <- gsub("\\OFPT_PACKET_IN", 1, opt_pkt_in)
opt_pkt_in_count <- replace_na(opt_pkt_in_count, "0")
opt_pkt_in_count <- as.numeric(opt_pkt_in_count)

opt_pkt_out <- str_extract(test$Info, "\\OFPT_PACKET_OUT")
opt_pkt_out_count <- gsub("\\OFPT_PACKET_OUT", 1, opt_pkt_out)
opt_pkt_out_count <- replace_na(opt_pkt_out_count, "0")
opt_pkt_out_count <- as.numeric(opt_pkt_out_count)

opt_flow_mod <- str_extract(test$Info, "\\OFPT_FLOW_MOD")
opt_flow_mod_count <- gsub("\\OFPT_FLOW_MOD", 1, opt_flow_mod)
opt_flow_mod_count <- replace_na(opt_flow_mod_count, "0")
opt_flow_mod_count <- as.numeric(opt_flow_mod_count)

opt_port_status <- str_extract(test$Info, "\\OFPT_PORT_STATUS")
opt_port_status_count <- gsub("\\OFPT_PORT_STATUS", 1, opt_port_status)
opt_port_status_count <- replace_na(opt_port_status_count, "0")
opt_port_status_count <- as.numeric(opt_port_status_count)

opt_feat_request <- str_extract(test$Info, "\\OFPT_FEATURES_REQUEST")
opt_feat_request_count <- gsub("\\OFPT_FEATURES_REQUEST", 1, opt_feat_request)
opt_feat_request_count <- replace_na(opt_feat_request_count, "0")
opt_feat_request_count <- as.numeric(opt_feat_request_count)

opt_set_cfg <- str_extract(test$Info, "\\OFPT_SET_CONFIG")
opt_set_cfg_count <- gsub("\\OFPT_SET_CONFIG", 1, opt_set_cfg)
opt_set_cfg_count <- replace_na(opt_set_cfg_count, "0")
opt_set_cfg_count <- as.numeric(opt_set_cfg_count)

opt_echo_req <- str_extract(test$Info, "\\OFPT_ECHO_REQUEST")
opt_echo_req_count <- gsub("\\OFPT_ECHO_REQUEST", 1, opt_echo_req)
opt_echo_req_count <- replace_na(opt_echo_req_count, "0")
opt_echo_req_count <- as.numeric(opt_echo_req_count)

opt_echo_rep <- str_extract(test$Info, "\\OFPT_ECHO_REPLY")
opt_echo_rep_count <- gsub("\\OFPT_ECHO_REPLY", 1, opt_echo_rep)
opt_echo_rep_count <- replace_na(opt_echo_rep_count, "0")
opt_echo_rep_count <- as.numeric(opt_echo_rep_count)

fin_ack <- str_extract(test$Info, "\\[FIN, ACK]")
fin_ack_count <- gsub("\\[FIN, ACK]", 1, fin_ack)
fin_ack_count <- replace_na(fin_ack_count, "0")
fin_ack_count <- as.numeric(fin_ack_count)

syn_ack <- str_extract(test$Info, "\\[SYN, ACK]")
syn_ack_count <- gsub("\\[SYN, ACK]", 1, syn_ack)
syn_ack_count <- replace_na(syn_ack_count, "0")
syn_ack_count <- as.numeric(syn_ack_count)

conf_root <- str_extract(test$Info, "\\Conf. Root ")
conf_root_count <- gsub("\\Conf. Root " , 1, conf_root)
conf_root_count <- replace_na(conf_root_count, "0")
conf_root_count <- as.numeric(conf_root_count)

echo_ping_req <- str_extract(test$Info, "\\Echo (ping) request ")
echo_ping_req_count <- gsub("\\Echo (ping) request ", 1, echo_ping_req)
echo_ping_req_count <- replace_na(echo_ping_req_count, "0")
echo_ping_req_count <- as.numeric(echo_ping_req_count)

echo_ping_rep <- str_extract(test$Info, "\\Echo (ping) reply ")
echo_ping_rep_count <- gsub("\\Echo (ping) reply ", 1, echo_ping_rep)
echo_ping_rep_count <- replace_na(echo_ping_rep_count, "0")
echo_ping_rep_count <- as.numeric(echo_ping_rep_count)

infor20 <- str_extract(test$Info, "\\Who has ")
info_converted20 <- gsub("\\Who has ", 20, infor20)
info_converted20 <- replace_na(info_converted20, "999")
info_converted20 <- as.numeric(info_converted20)

conf_tc_root <- str_extract(test$Info, "\\Conf. TC + Root ") 
conf_tc_root_count <- gsub("\\Conf. TC + Root ", 1, conf_tc_root)
conf_tc_root_count <- replace_na(conf_tc_root_count, "0")
conf_tc_root_count <- as.numeric(conf_tc_root_count)

std_query <- str_extract(test$Info, "\\Standard query ") 
std_query_count <- gsub("\\Standard query ", 1, std_query)
std_query_count <- replace_na(std_query_count, "0")
std_query_count <- as.numeric(std_query_count)

tcp_retrans <- str_extract(test$Info, "\\[TCP Retransmission]")
tcp_retrans_count <- gsub("\\[TCP Retransmission]", 1, tcp_retrans)
tcp_retrans_count <- replace_na(tcp_retrans_count, "0")
tcp_retrans_count <- as.numeric(tcp_retrans_count)

tcp_fst_retrans <- str_extract(test$Info, "\\[TCP Fast Retransmission]")
tcp_fst_retrans_count <- gsub("\\[TCP Fast Retransmission]", 1, tcp_fst_retrans)
tcp_fst_retrans_count <- replace_na(tcp_fst_retrans_count, "0")
tcp_fst_retrans_count <- as.numeric(tcp_fst_retrans_count)

Length <- test$Length
number <- test$No.

Label <- c()
i <- 0
for (i in 1:length(test$Info)) {
   if (ack_count[i] == 1) {
	Label[i] <- 1
   } 
   else if (rst_ack_count[i] == 1) {
   	Label[i] <- 1
   }
   else if ((syn_count[i] == 1) && (syn_ack_count[i+1] == 1)) {
   	Label[i] <- 1
   }
   else if (protocol_converted[i] == 6006) {
   	Label[i] <- 1
   }
   else if (echo_ping_req_count[i] == 1) {
   	Label[i] <- 1
   }
   else if (echo_ping_rep_count[i] == 1) {
   	Label[i] <- 1
   }
   else if (opt_pkt_in_count[i] == 1) {
   	Label[i] <- 1
   }
   else if (opt_pkt_out_count[i] == 1) {
 	Label[i] <- 1
   }
   else if (tcp_dup_ack_count[i] == 1) {
   	Label[i] <- 1
   }
   else if (tcp_retrans_count[i] == 1) {
   	Label[i] <- 1
   }
   else if (rst_count[i] == 1) {
   	Label[i] <- 1
   }
   else if (tcp_fst_retrans_count[i] == 1) {
   	Label[i] <- 1
   }
   else
     Label[i] <- 0
}

# write out to a new csv file
convert_data <- data.frame(number, time, Source_address, Destination_address, protocol_converted, Length, ack_count, syn_count, rst_ack_count, psh_count, rst_count, tcp_dup_ack_count, opt_hello_count, opt_pkt_in_count, opt_pkt_out_count, opt_flow_mod_count, opt_port_status_count, opt_feat_request_count, opt_set_cfg_count, opt_echo_req_count, opt_echo_rep_count, fin_ack_count, syn_ack_count, echo_ping_req_count, echo_ping_rep_count, conf_tc_root_count, std_query_count, Label)
write.csv(convert_data, output)
print("Done with data preprocessing")