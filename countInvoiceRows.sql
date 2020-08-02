USE dataaid;

SELECT * FROM lineitems
WHERE InvoiceNo = 82729877;

SELECT AVG(UnitPrice)
FROM lineitems;

SELECT count(*)
FROM invoices
where CustomerId =1;

SELECT count(*)
FROM lineitems;

select CustomerID, Email
FROM users;

SELECT *, MIN(UnitPrice)
FROM lineitems
GROUP BY LineItemId;

SELECT *
FROM lineitems
ORDER BY UnitPrice DESC;