#getwd()
#setwd("Desktop/Data-Aid")
library(dplyr)
library(tidyverse)
library(readxl)
library(knitr)
library(ggplot2)
library(lubridate)
library(arules)
library(arulesViz)
library(plyr)



retail = read.csv("OnlineRetail.csv")
summary(retail)
nrow(retail)
retail = retail[complete.cases(retail),]
retail = retail %>% mutate(Description = as.factor(Description))
retail = retail %>% mutate(Country= as.factor(Country))
retail$Date = as.Date(retail$InvoiceDate)
retail$Time = str_sub(as.character(retail$InvoiceDate),-5,-1)
retail$Time = paste0(retail$Time,":00")
retail$InvoiceNo = as.numeric(as.character(retail$InvoiceNo))

glimpse(retail)


retail$Time = as.factor(retail$Time)
a = hms(as.character(retail$Time))
retail$Time = hour(a)

#### What time do people often purchase online?
retail %>%
  ggplot(aes(x=Time)) +
  geom_histogram(stat = "count", fill = "indianred")

#### How many items did each customer buy?
detach("package:plyr", unload = TRUE)

retail %>%
  group_by(InvoiceNo) %>%
  summarize(n_items = mean(Quantity)) %>%
  ggplot(aes(x=n_items)) +
  geom_histogram(fill="indianred", bins = 100000) +
  xlab('Number of Items') + ylab('Number of Customers') +
  ggtitle('Number of Customers Who Bought N number of Items')+
  coord_cartesian(xlim = c(0,80))

##### Top 10 best Sellers?
tmp = retail %>%
  group_by(StockCode, Description) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
tmp = head(tmp, n=10)
tmp

tmp %>%
  ggplot(aes(x=reorder(Description, count), y =count)) +
  geom_bar(stat = "identity", fill="grey")+
  coord_flip()


###### Association Rules ##########

## 1. We need to transform data from data.frame to transactions
### such that we have items bought together in one row

retail_sorted = retail[order(retail$CustomerID),]
library(plyr)
itemList = ddply(retail,c("CustomerID","Date"),
                 function(df1)paste(df1$Description, collapse = ','))

## ddply accepts a dataframe, splits into peices
## based on one or more factors
## computes on the peices
## returns a dataframe
## use , to seperate different items

itemList$CustomerID = NULL
itemList$Date = NULL
colnames(itemList) = c("items")

##write the data into a csv file
write.csv(itemList,"market_basket.csv", quote = FALSE, row.names = TRUE)

tr = read.transactions('market_basket.csv', format = 'basket', sep = ',')
tr
summary(tr)

### density: % of non-empty cells in the sparse matrix

## item frequency plot
itemFrequencyPlot(tr, topN = 20, type='absolute')


#### create some rules

rules = apriori(tr, parameter = list(supp=0.001, conf =0.8, maxlen=100))
rules = sort(rules, by = 'confidence', decreasing = TRUE)
summary(rules)
inspect(rules[1:10])


##plot top 10 rules
topRules = rules[1:10]
plot(topRules)

plot(topRules, method = 'graph')

plot(topRules, method = 'grouped')



rules = sort(rules, by = 'support', decreasing=TRUE)
  
