

# This is a program to predict the stock price in the ugandan stock market
import csv
import requests
from bs4 import BeautifulSoup
from csv import writer
import time
# Getting data from Uganda Securities Exchange
url='https://www.use.or.ug/content/market-snapshot'
response = requests.get(url)
soup = BeautifulSoup(response.text,'html.parser')

# Assigning a name to the data
filename ='use data.csv'
csv_writer = csv.writer(open(filename,'w'))

# Making sure that the file gets updated by using time as a constraint
timestr1 = time.strftime("%Y%m%d-%H%M%S")
changed_filename = filename.split(".")[0]+timestr1+"."+filename.split(".")[1]
print(changed_filename)

# Getting the data from the table in the U.S.E site ie. Market snapshot
for tr in soup.find_all('tr'):
    data =[]

    for th in tr.find_all('th'):
        data.append(th.text)

    if data:
        print("Inserting Headers:{}".format(','.join(data)))
        csv_writer.writerow(data)
        continue

    for td in tr.find_all('td'):
        data.append(td.text.strip())

    if data:
        print("Inserting Table Data:{}".format(','.join(data)))
        csv_writer.writerow(data)



















































