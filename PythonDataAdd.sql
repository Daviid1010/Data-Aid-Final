use dataaid;
SET FOREIGN_KEY_CHECKS = 0;

LOAD DATA LOCAL INFILE '/home/david/Desktop/Data-Aid/PythonInvoiceItems.csv'
INTO TABLE Invoices
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA LOCAL INFILE '/home/david/Desktop/Data-Aid/PythonLineItem.csv'
REPLACE INTO TABLE lineitems
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
