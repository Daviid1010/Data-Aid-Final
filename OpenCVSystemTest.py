import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
try:
    from PIL import Image
except ImportError:
    import Image

import pandas as pd
from datetime import datetime




out = pytesseract.image_to_string(cv2.imread("Images/schuh.jpeg"))


####P Print Out of Raw Text
print(out)


#### Split Data into Arrays
arrays = out.split("\n\n")
data = out.split("\n")
print("Data: ",data)
arrays.remove(" ")
arrays.remove("  ")
print("Arrays: ",arrays)
splitArrays = arrays



###### Refactoring for Database Entry (not dynamic yet!!!)
InvoiceNum = arrays[5]
shop = arrays[0]
print(shop)
shopAddress1 = arrays[1]
shopAddress2 = arrays[2]
print("Shop Address 1: ",shopAddress1)
print("Shop Address 2: ",shopAddress2)

print("VAT Number:",data[6][-8:])



InvoiceNum = int(data[6][-8:])

date = datetime.date(datetime.now())
dateStr = date.strftime("%d/%m/%Y")
print(dateStr)

item1 = arrays[5]
print("Item 1: "+item1)
split1 = item1.split("\n");
print(split1)
line1desc = split1[0]
line1price = float(split1[6])
line1quantity = 1


print("####### Item 1")
print(line1desc)
print(line1price)
print(line1quantity)



item2 = arrays[6]
print("Item 2: "+item2)
split2 = item2.split("\n")
print(split2)
line2desc = split2[1]
line2price = float(split2[4][-5:])
quantityline2 = 1

print("###### Item 2")
print(line2desc)
print(line2price)
print(quantityline2)





item3 = arrays[7]
print("Item 3: "+item3)
split3 = item3.split("\n")
print(split3)
item3desc = split3[0]


split4 = arrays[8]
print(split4)
item3price = float(split4[-4:])
item3quantity = 1

print("############ Item 3 ")
print(item3desc)
print(item3price)
print(item3quantity)



dataSetLineItems = pd.DataFrame({
    'InvoiceNo':[InvoiceNum,InvoiceNum,InvoiceNum],
    'Description':[line1desc,line2desc,item3desc],
    'UnitPrice:':[line1price,line2price,item3price],
    'Quantity':[line1quantity,quantityline2,item3quantity]
})

print(dataSetLineItems)


dataSetInvoice= pd.DataFrame({
    'InvoiceNo':[InvoiceNum],
    'InvoiceDate':[dateStr],
    'location': [shopAddress1],
    'Country':["Ireland"],
    'CustomerId':[1]
})

print(dataSetInvoice)

dataSetLineItems.to_csv("PythonLineItem.csv", sep=',')
dataSetInvoice.to_csv("PythonInvoiceItems.csv", sep=',')


