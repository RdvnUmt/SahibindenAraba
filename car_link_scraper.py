from selenium import webdriver
import chromedriver_autoinstaller
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import json
import time

chromedriver_autoinstaller.install()
driver = webdriver.Chrome()
car_link_set = set()


#Araba modelleri belirlenmeli 2500 üstü
car_model_list = ["bmw","audi","citroen","fiat","ford","honda","hyundai","mercedes-benz",
                  "opel","peugeot","renault","seat","skoda","tofas","toyota",
                  "volkswagen"]

# +2.5K 16 models
# +1K - 2.5K chevrolet, dacia, kia, nissan, volvo  5 model eklenmeli mi eklenmemeli mi

for model in car_model_list:

    starting_time = time.time()

    print(f"Model {model} için tarama başladı...")
    for i in range(1,3): #1,51 - 50 sayfa değerlendirilicek

        url = f"https://www.arabam.com/ikinci-el/otomobil/{model}?take=50&page={i}"
        driver.get(url)

        
        table_element = driver.find_element(By.XPATH,"//table[@id='main-listing']")

        car_list = WebDriverWait(driver, 5).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'tr'))
        )

        for car_item in car_list:
            try:
                car_link  = car_item.find_element(By.TAG_NAME,'a').get_attribute('href')
                car_link_set.add(car_link)
                count = count +1
            except:
                print("Tag a bulunamadı maalesef")    

        filename = f"car_links.json"

        car_link_arr = list(car_link_set)

        with open(filename,'w',encoding="utf-8") as file:
            json.dump(car_link_arr,file,ensure_ascii=False, indent=4)
            time.sleep(10)

        finish_time = time.time()

        print(f"Model {model} için harcanan süre {finish_time - starting_time}s")



driver.quit()


