from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time

# Load locations from CSV
loc = pd.read_csv("D:/Projects/app_streamlit/datasets/loc.csv")
loc = loc['loc'].tolist()

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

# Open Google
chrome_driver.get('https://www.google.com/')
time.sleep(1)

# Create an empty DataFrame to store the results
df = pd.DataFrame(columns=['location', 'link'])

i = 0

while i < len(loc):
    # Fetch the search input box using xpath
    search_box = chrome_driver.find_element(By.XPATH, '//*[@id="APjFqb"]')
    search_box.clear()
    search_box.send_keys(loc[i] + ' 99acres')
    time.sleep(1)

    search_box.send_keys(Keys.ENTER)
    time.sleep(6)

    html = chrome_driver.page_source
    soup = BeautifulSoup(html, 'lxml')

    # Find the first 'a' tag with the given jsname
    a_tag = soup.find('a', jsname="UWckNb")
    if a_tag and 'href' in a_tag.attrs:
        href = a_tag['href']
    else:
        href = None  # Assign None if no href is found

    # Create a DataFrame for the current row
    new_row = pd.DataFrame({'location': [loc[i]], 'link': [href]})

    # Concatenate the new row to the existing DataFrame
    df = pd.concat([df, new_row], ignore_index=True)

    # Save the DataFrame to a CSV file after each iteration
    df.to_csv('D:/Projects/app_streamlit/datasets/data_link.csv', index=False)

    i += 1
    print(href)

    # Pause every 15 iterations to avoid being blocked
    if i % 15 == 0:
        time.sleep(5)

# Close the browser
chrome_driver.quit()

# Print the DataFrame to check the results
print(df)
