from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time

df = pd.read_csv("datasets\location.csv")
loc = df.location.tolist() 

# Define the driver path
driver_path = Service("C:/chromedriver.exe")

# Set the different options for the browser
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

# Ignore the certificate and SSL errors
chrome_options.add_argument('--ignore-certificate-errors')
chrome_options.add_argument('--ignore-ssl-errors')
                             
# Maximize the browser window
chrome_options.add_argument("start-maximized")

# Define the driver and open the browser
chrome_driver = webdriver.Chrome(service=driver_path, options=chrome_options)

chrome_driver.get('http://google.com')
time.sleep(1)

# fetch the search input box using xpath
user_input = chrome_driver.find_element(by=By.XPATH, value='//*[@id="APjFqb"]')
user_input.send_keys('India latitude and longitude')
time.sleep(1)

user_input.send_keys(Keys.ENTER)
time.sleep(1)

latlong = []
i = 0
while i < len(loc):
    # fetch the search input box using xpath
    chrome_driver.find_element(by=By.XPATH, value='//*[@id="APjFqb"]').clear()
    user_input = chrome_driver.find_element(by=By.XPATH, value='//*[@id="APjFqb"]')
    user_input.send_keys(loc[i] + ' latitude and longitude')
    time.sleep(1)   

    user_input.send_keys(Keys.ENTER)
    time.sleep(4)   

    html = chrome_driver.page_source
    soup = BeautifulSoup(html,'lxml')    

    if soup.find('div', class_='Z0LcW t2b5Cf'):
        latlong.append(soup.find('div', class_='Z0LcW t2b5Cf').text.strip())
    elif soup.find("div", class_="Z0LcW t2b5Cf vMhfn"):
        latlong.append(soup.find("div", class_="Z0LcW t2b5Cf vMhfn").text.strip())    
    elif soup.find("span", class_="hgKElc"):
        latlong.append(soup.find("span", class_="hgKElc").text.strip())  
    else:
        latlong.append(np.NaN)

    i+=1

    if i%15==0:
        time.sleep(5)

print(latlong)
df=pd.DataFrame({'location':loc,
  'coordinates':latlong
  })

df.to_csv('lat_long.csv', index=False)    