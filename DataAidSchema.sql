USE dataaid;

DROP TABLE IF EXISTS Users;

CREATE TABLE Users(
	CustomerId INT AUTO_INCREMENT,
    email VARCHAR(200) NOT NULL,
    pword VARCHAR(200) NOT NULL,
    PRIMARY KEY(CustomerId)
);

DROP TABLE IF EXISTS Invoices;

CREATE TABLE Invoices(
	InvoiceId INT auto_increment,
    InvoiceNo INT UNIQUE,
    InvoiceDate VARCHAR(255),
    Location VARCHAR(255),
    Country VARCHAR(255),
	CustomerId INT,
    PRIMARY KEY(InvoiceId),
    FOREIGN KEY (CustomerId) REFERENCES Users(CustomerId)
);

DROP TABLE IF EXISTS LineItems;

CREATE TABLE LineItems(
	LineItemId INT AUTO_INCREMENT,
    InvoiceNo INT,
    Description VARCHAR(500),
    UnitPrice DECIMAL,
    Quantity INT,
    PRIMARY KEY(LineItemId),
    FOREIGN KEY(InvoiceNo) REFERENCES Invoices(InvoiceNo)
);
