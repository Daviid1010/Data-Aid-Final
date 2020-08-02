    ##### R Code which grabs data from the Kaggle DataSet and UCI Dataset on Retail Sales Data
    ##### This code preproccesses the data for the purpose of the mid-point presentation prototype
    ##### Sources:
    # UCI: http://archive.ics.uci.edu/ml/datasets/online+retail
    # Kaggle: https://www.kaggle.com/aungpyaeap/supermarket-sales/activity
    
    
    getwd()
    setwd("Desktop/Data-Aid")
    getwd()
    
    ## install and add R packages
    ##install.packages("dpylr")
    ##install.packages("gdata")
    ##install.packages("xlsx")
    ##install.packages("rio")
    ##install_formats()
    library(rio)
    library(gdata)
    library(dplyr)
    library(xlsx)
    ## reading the csv file from kaggle
    kaggle = read.csv(file = "Datasets/kaggle_supermarket_dataset.csv")
    head(kaggle)
    attributes(kaggle)
    nrow(kaggle)
    summary(kaggle)
    
    ## removing uneeded columns from data
    columnsRM = c("Customer.type","Gender","cogs","gross.margin.percentage","gross.income","Rating", "Payment")
    kaggle = kaggle[!names(kaggle) %in% columnsRM]
    head(kaggle)
    
    #### Test For Normality
    print(max(kaggle$Total))
    print(min(kaggle$Total))
    hist(kaggle$Total)
    shapiro.test(kaggle$Total)
    
    ## data refactoring
    unique(kaggle$Invoice.ID)
    kaggle$Invoice.ID = as.character(kaggle$Invoice.ID)
    kaggle$City = as.character(kaggle$City)
    kaggle$Date = as.character(kaggle$Date)
    kaggle$Time = as.character(kaggle$Time)
    kaggle$Country = "Myanmar"
    
    ?gsub
    ## change Invoice Id to a numeric value
    kaggle$Invoice.ID = sapply(kaggle$Invoice.ID, function(x) { gsub("-","",x)})
    unique(kaggle$Invoice.ID)
    length(unique(kaggle$Invoice.ID))
    ## covert char invoice id to numeric
    kaggle$Invoice.ID = as.numeric(kaggle$Invoice.ID)
    head(kaggle)
    
    ## combine city and branch to make location and store
    kaggle$Branch = as.character(kaggle$Branch)
    kaggle$location = paste(kaggle$City," ",kaggle$Branch)
    columnsRM = c("City","Branch")
    kaggle = kaggle[!names(kaggle) %in% columnsRM]
    
    
    ##Rename Product Line as Description in Kaggle Dataset
    kaggle$Product.line = as.character(kaggle$Product.line)
    
    
    #### get unit price including tax and convert from MMR to EUR
    kaggle$Unit.price = kaggle$Unit.price + (kaggle$Tax.5. / kaggle$Quantity)
    
    
    columnsRM = c("Tax.5.","Total")
    kaggle = kaggle[!names(kaggle) %in% columnsRM]
    
    ##add time and date together, make it consistent with uci
    
    kaggle$DateTime = paste(as.character(kaggle$Date),"T",kaggle$Time,"Z")
    kaggle$DateTime = sapply(kaggle$DateTime, function(x) { gsub(" ","",x)})
    
    ## sperate out data into two dataframes
    
    kaggleInvoice = kaggle %>%
      select(Invoice.ID, DateTime, Country, location)
    
    names(kaggleInvoice)[names(kaggleInvoice) == "Invoice.ID"] <- "InvoiceNo"
    names(kaggleInvoice)[names(kaggleInvoice) == "DateTime"] <- "InvoiceDate"
    
    length(unique(kaggleInvoice$InvoiceNo))
    
    ## 1000 unique instances in kaggle dataset, 10000 instances, no need to check for duplicate invoices
    ### we can safely remove data from kaggle
    columnsRM = c("Date","Time", "DateTime","Country","Location")
    kaggle = kaggle[!names(kaggle) %in% columnsRM]
    
    ##rename data to be consistent with uci
    names(kaggle)[names(kaggle) == "Product.line"] <- "Description"
    names(kaggle)[names(kaggle) == "Invoice.ID"] <- "InvoiceNo"
    names(kaggle)[names(kaggle) == "Unit.price"] <- "Unit Price"
    names(kaggle)[names(kaggle) == "Unit Price"] <- "UnitPrice"
    
    #######################################
    
    ##UCI Dataset
    convert("uciOnlineRetail.xlsx","uciOnlineRetail.csv")
    uci = read.csv("OnlineRetail.csv")
    nrow(uci)
    head(uci)
    summary(uci)
    attributes(uci)
    unique(uci$InvoiceNo)
    summary(uci)
    sum(is.na(uci$InvoiceNo))
    sum(!complete.cases(uci))

    
    hist(uci$UnitPrice)
    max(uci$UnitPrice)
    min(uci$UnitPrice)
    boxplot(uci$UnitPrice)
    mean(uci$UnitPrice)
    
    ## uci data refactor and removal of some rows
    uci$InvoiceNo = as.character(uci$InvoiceNo)
    uci$InvoiceDate = as.character(uci$InvoiceDate)
    uci = uci[!names(uci) %in% c("CustomerID")]
    
    length(unique(uci$InvoiceNo))
    length(uci)
    
    ## make all chars number in invpice id
    uci$InvoiceNo = sapply(uci$InvoiceNo, function(x) { gsub("[^0-9\\.]", "", x) })
    uci$InvoiceNo = as.numeric(uci$InvoiceNo)
    ## 22061 unique invoices in 541909 instances
    length(unique(uci$InvoiceNo))
    head(uci, n=100)
    tail(uci)
    
    ##### Some Tests on UCI Date
    print(mean(uci$UnitPrice))
    print(min(uci$UnitPrice))
    print(max(uci$UnitPrice))
    hist(uci$UnitPrice)
    boxplot(uci$UnitPrice)
    print(unique(uci$Description))
  /##### 4212 Unique Values in 541909 instances
    #####
    
    ## create new data frane with invoice data, need to remove dupliacte rows
    uciInvoice = uci %>%
      select(InvoiceNo, InvoiceDate, Country)
    
    head(uciInvoice)
    tail(uciInvoice)
    duplicated(uciInvoice$InvoiceNo)
    uciInvoice = uciInvoice[!duplicated(uciInvoice$InvoiceNo),]
    length(unique(uciInvoice$InvoiceNo))
    sum(is.na(uciInvoice$InvoiceNo))
    ## remove uci invoice columns
    columnsRM = c("StockCode","InvoiceDate","Country")
    uci = uci[!names(uci) %in% columnsRM]
    
    
    ##add blank location column (online data so no location)
    uciInvoice$Location = ""
    
    ##chnage country to character
    uciInvoice$Country = as.character(uciInvoice$Country)
    uci$Description = as.character(uci$Description)
    
    
    
    ################################
    
    ##merge the data on the two dataframes
    tail(kaggleInvoice)
    tail(uciInvoice)
    summary(uciInvoice)
    InvoiceAll = rbind(kaggleInvoice,uciInvoice)
    LineAll = rbind(kaggle,uci)
    names(uci)
    names(kaggle)
    
    ##### addd one as foreign key to all line items
    InvoiceAll$CustomerId = 1
    
    
    head(InvoiceAll)
    head(LineAll)
    tail(LineAll)
    tail(InvoiceAll)
    ## write uci data to csv in two tables
    write.csv(file = "UCIINvoice.csv",uciInvoice)
    write.csv(file = "UCILineItems.csv",uci)  
    
    write.csv(file ="InvoiceData.csv",InvoiceAll)
    write.csv(file = "LineItemData.csv",LineAll)
    
      
