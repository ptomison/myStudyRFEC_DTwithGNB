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

#   read in dataset
test <- read.csv(test, header = TRUE, na.strings = "")
head(test)
nrow(test)

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

#MS-NLB-PhysServer-32_7f:cb:c9:96
address <- gsub("\\MS-NLB-PhysServer-32_7f:cb:c9:96", "32:7f:cb:c9:96", address) 
#MS-NLB-PhysServer-16_da:13:20:25
address <- gsub("\\MS-NLB-PhysServer-16_da:13:20:25", "16:da:13:20:25", address) 
#MS-NLB-PhysServer-23_b2:04:ec:4b
address <- gsub("\\MS-NLB-PhysServer-23_b2:04:ec:4b", "23:b2:04:ec:4b", address) 
#MS-NLB-PhysServer-31_a0:85:9b:01
address <- gsub("\\MS-NLB-PhysServer-31_a0:85:9b:01", "31:a0:85:9b:01", address) 

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
infor1 <- str_extract(test$Info, "\\[ACK]")
info_converted1 <- gsub("\\[ACK]", 1, infor1)
info_converted1 <- replace_na(info_converted1, "999")
info_converted1 <- as.numeric(info_converted1)

infor2 <- str_extract(test$Info, "\\[SYN]")
info_converted2 <- gsub("\\[SYN]", 2, infor2)
info_converted2 <- replace_na(info_converted2, "999")
info_converted2 <- as.numeric(info_converted2)

infor3 <- str_extract(test$Info, "\\[RST, ACK]")
info_converted3 <- gsub("\\[RST, ACK]", 3, infor3)
info_converted3 <- replace_na(info_converted3, "999")
info_converted3 <- as.numeric(info_converted3)

infor4 <- str_extract(test$Info, "\\[PSH]")
info_converted4 <- gsub("\\[PSH]", 4, infor4)
info_converted4 <- replace_na(info_converted4, "999")
info_converted4 <- as.numeric(info_converted4)

infor5 <- str_extract(test$Info, "\\TCP Dup ACK ") 
info_converted5 <- gsub("\\TCP Dup ACK ", 5, infor5)
info_converted5 <- replace_na(info_converted5, "999")
info_converted5 <- as.numeric(info_converted5)

infor6 <- str_extract(test$Info, "\\OFPT_HELLO")
info_converted6 <- gsub("\\OFPT_HELLO", 6, infor6)
info_converted6 <- replace_na(info_converted6, "999")
info_converted6 <- as.numeric(info_converted6)

infor7 <- str_extract(test$Info, "\\OFPT_PACKET_IN")
info_converted7 <- gsub("\\OFPT_PACKET_IN", 7, infor7)
info_converted7 <- replace_na(info_converted7, "999")
info_converted7 <- as.numeric(info_converted7)

infor8 <- str_extract(test$Info, "\\OFPT_PACKET_OUT")
info_converted8 <- gsub("\\OFPT_PACKET_OUT", 8, infor8)
info_converted8 <- replace_na(info_converted8, "999")
info_converted8 <- as.numeric(info_converted8)

infor9 <- str_extract(test$Info, "\\OFPT_FLOW_MOD")
info_converted9 <- gsub("\\OFPT_FLOW_MOD", 9, infor9)
info_converted9 <- replace_na(info_converted9, "999")
info_converted9 <- as.numeric(info_converted9)

infor10 <- str_extract(test$Info, "\\OFPT_PORT_STATUS")
info_converted10 <- gsub("\\OFPT_PORT_STATUS", 10, infor10)
info_converted10 <- replace_na(info_converted10, "999")
info_converted10 <- as.numeric(info_converted10)

infor11 <- str_extract(test$Info, "\\OFPT_FEATURES_REQUEST")
info_converted11 <- gsub("\\OFPT_FEATURES_REQUEST", 11, infor11)
info_converted11 <- replace_na(info_converted11, "999")
info_converted11 <- as.numeric(info_converted11)

infor12 <- str_extract(test$Info, "\\OFPT_SET_CONFIG")
info_converted12 <- gsub("\\OFPT_SET_CONFIG", 12, infor12)
info_converted12 <- replace_na(info_converted12, "999")
info_converted12 <- as.numeric(info_converted12)

infor13 <- str_extract(test$Info, "\\OFPT_ECHO_REQUEST")
info_converted13 <- gsub("\\OFPT_ECHO_REQUEST", 13, infor13)
info_converted13 <- replace_na(info_converted13, "999")
info_converted13 <- as.numeric(info_converted13)

infor14 <- str_extract(test$Info, "\\OFPT_ECHO_REPLY")
info_converted14 <- gsub("\\OFPT_ECHO_REPLY", 14, infor14)
info_converted14 <- replace_na(info_converted14, "999")
info_converted14 <- as.numeric(info_converted14)

infor15 <- str_extract(test$Info, "\\[FIN, ACK]")
info_converted15 <- gsub("\\[FIN, ACK]", 15, infor15)
info_converted15 <- replace_na(info_converted15, "999")
info_converted15 <- as.numeric(info_converted15)

infor16 <- str_extract(test$Info, "\\[SYN, ACK]")
info_converted16 <- gsub("\\[SYN, ACK]", 16, infor16)
info_converted16 <- replace_na(info_converted16, "999")
info_converted16 <- as.numeric(info_converted16)

infor17 <- str_extract(test$Info, "\\Conf. Root ")
info_converted17 <- gsub("\\Conf. Root " , 17, infor17)
info_converted17 <- replace_na(info_converted17, "999")
info_converted17 <- as.numeric(info_converted17)

infor18 <- str_extract(test$Info, "\\Echo (ping) request ")
info_converted18 <- gsub("\\Echo (ping) request ", 18, infor18)
info_converted18 <- replace_na(info_converted18, "999")
info_converted18 <- as.numeric(info_converted18)

infor19 <- str_extract(test$Info, "\\Echo (ping) reply ")
info_converted19 <- gsub("\\Echo (ping) reply ", 19, infor19)
info_converted19 <- replace_na(info_converted19, "999")
info_converted19 <- as.numeric(info_converted19)

infor20 <- str_extract(test$Info, "\\Who has ")
info_converted20 <- gsub("\\Who has ", 20, infor20)
info_converted20 <- replace_na(info_converted20, "999")
info_converted20 <- as.numeric(info_converted20)

infor21 <- str_extract(test$Info, "\\Conf. TC + Root ") 
info_converted21 <- gsub("\\Conf. TC + Root ", 21, infor21)
info_converted21 <- replace_na(info_converted21, "999")
info_converted21 <- as.numeric(info_converted21)

infor22 <- str_extract(test$Info, "\\Standard query ") 
info_converted22 <- gsub("\\Standard query ", 22, infor22)
info_converted22 <- replace_na(info_converted22, "999")
info_converted22 <- as.numeric(info_converted22)

# Now combine the 22 information arrays into a single array 
info_converted <- c()
i <- 0
for (i in 1:length(test$Info)) {
   if (info_converted1[i] == 1) {
	info_converted <- c(info_converted, info_converted1[i])  # Append matching elements to the result
   } 
   else if (info_converted2[i] == 2) {
   	info_converted <- c(info_converted, info_converted2[i])
   }
   else if (info_converted3[i] == 3) {
   	info_converted <- c(info_converted, info_converted3[i])
   }
   else if (info_converted4[i] == 4) {
   	info_converted <- c(info_converted, info_converted4[i])
   }
   else if (info_converted5[i] == 5) {
   	info_converted <- c(info_converted, info_converted5[i])
   }
   else if (info_converted6[i] == 6) {
   	info_converted <- c(info_converted, info_converted6[i])
   }
   else if (info_converted7[i] == 7) {
   	info_converted <- c(info_converted, info_converted7[i])
   }
   else if (info_converted8[i] == 8) {
   	info_converted <- c(info_converted, info_converted8[i])
   }
   else if (info_converted9[i] == 9) {
   	info_converted <- c(info_converted, info_converted9[i])
   }
   else if (info_converted10[i] == 10) {
   	info_converted <- c(info_converted, info_converted10[i])
   }
   else if (info_converted11[i] == 11) {
   	info_converted <- c(info_converted, info_converted11[i])
   }
   else if (info_converted12[i] == 12) {
   	info_converted <- c(info_converted, info_converted12[i])
   }
   else if (info_converted13[i] == 13) {
   	info_converted <- c(info_converted, info_converted13[i])
   }
   else if (info_converted14[i] == 14) {
   	info_converted <- c(info_converted, info_converted14[i])
   }
   else if (info_converted15[i] == 15) {
   	info_converted <- c(info_converted, info_converted15[i])
   }
   else if (info_converted16[i] == 16) {
   	info_converted <- c(info_converted, info_converted16[i])
   }
   else if (info_converted17[i] == 17) {
   	info_converted <- c(info_converted, info_converted17[i])
   }
   else if (info_converted18[i] == 18) {
   	info_converted <- c(info_converted, info_converted18[i])
   }
   else if (info_converted19[i] == 19) {
   	info_converted <- c(info_converted, info_converted19[i])
   }
   else if (info_converted20[i] == 20) {
   	info_converted <- c(info_converted, info_converted20[i])
   }
   else if (info_converted21[i] == 21) {
   	info_converted <- c(info_converted, info_converted21[i])
   }
   else if (info_converted22[i] == 22) {
   	info_converted <- c(info_converted, info_converted22[i])
   }
   else
     info_converted <- c(info_converted, 999)
}

Length <- test$Length
number <- test$No.

# write out to a new csv file
convert_data <- data.frame(number, time, Source_address, Destination_address, protocol_converted, Length, info_converted)
write.csv(convert_data, "test_convert.csv")
print("Done with data preprocessing")
