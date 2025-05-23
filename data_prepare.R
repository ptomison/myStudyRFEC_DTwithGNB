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

if(!require(iptools)){
	install.packages("iptools")
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


setwd("C:/PhD/DIS9903A/Week 7")
getwd()

#   read in dataset
test <- read.csv("test.csv", header = TRUE, na.strings = "")
time = test$Time
address = ip_address(test$Source)
address <- gsub("\\NA", "99.99.99.99", address)
address <- replace_na(address, "99.99.99.99")
Source_address <- gsub("\\.", "", address)

destination = ip_address(test$Destination)
destination <- gsub("\\NA", "99.99.99.99", destination )
destination <- replace_na(destination , "99.99.99.99")
Destination_address <- gsub("\\.", "", destination )

protocol = test$Protocol
protocol <- gsub("\\DNS", 1, protocol )
protocol <- gsub("\\TCP", 2, protocol )
protocol <- gsub("\\ARP", 3, protocol )
protocol <- gsub("\\OpenFlow", 4, protocol )
protocol <- gsub("\\M1", 5, protocol )

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

# Now combine the 15 arrays into a single array 
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
   else
     info_converted <- c(info_converted, 0)
}
Length <- test$Length
number <- test$No.

# write out to a new csv file
convert_data <- data.frame(number, time, Source_address, Destination_address, protocol, Length, info_converted)
write.csv(convert_data, "test_convert.csv")
print("Done with data preprocessing")