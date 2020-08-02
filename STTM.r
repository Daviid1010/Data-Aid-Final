install.packages("udpipe")
install.packages("tidyverse")
install.packages("BTM")

library(BTM)
library(udpipe)
library(tidyverse)

data("brussels_reviews_anno", package = "udpipe")

## Taking only nouns of Dutch data
x <- subset(brussels_reviews_anno, language == "nl")
x <- subset(x, xpos %in% c("NN", "NNP", "NNS"))
x <- x[, c("doc_id", "lemma")]

## Building the model
set.seed(321)
model  <- BTM(x, k = 5, beta = 0.01, iter = 1000, trace = 100)

## Inspect the model - topic frequency + conditional term probabilities
model$theta
topicterms <- terms(model)
topicterms

getwd()
setwd("Desktop/Data-Aid")

retail = read.csv("OnlineRetail.csv")


keeps = c("StockCode","Description")

retailNew = retail[, names(retail) %in% keeps]
distinct(retailNew$StockCode)
retailNew$StockCode = as.character(retailNew$StockCode)
retailNew$Description = as.character(retailNew$Description)

retailNew = retailNew[!duplicated(retailNew[,c('StockCode',"Description")]),]

s = strsplit(retailNew$Description, split = " ")


library(spacyr)
spacy_install()
spacy_initialize()
spacyr::spacy_extract_nounphrases(s)

newData = data.frame(StockCode = rep(retailNew$StockCode, sapply(s, length)), words = unlist(s))
retailNew$Description = spacyr::spacy_extract_nounphrases(retailNew$Description)$text

set.seed(2344)
model = BTM(newData, k =5, beta = 0.01, iter = 3000, trace = 100)

??BTM

model$theta
topicterms <- terms(model)
topicterms
