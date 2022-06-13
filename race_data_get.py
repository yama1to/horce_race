#https://db.netkeiba.com/race/200803010601/

"""
race_urlディレクトリに含まれるURLを利用して、htmlを取得する
"""
import datetime
import pytz
now_datetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

import requests
from bs4 import BeautifulSoup

import time
import os
from os import path
OWN_FILE_NAME = path.splitext(path.basename(__file__))[0]
RACR_URL_DIR = "race_url"
RACR_HTML_DIR = "race_html"


import logging
logger = logging.getLogger(__name__) #ファイルの名前を渡す

from tqdm import tqdm


from concurrent.futures import ThreadPoolExecutor
import requests
from functools import partial


def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def get_race_html():
    # 去年までのデータ
    for year in range(2008, now_datetime.year):
        for month in tqdm(range(1, 13)):
            #print(year,month)
            get_race_html_by_year_and_mon(year,month)
        # mapfunc = partial(requests.get, headers={'User-Agent': 'hoge'})
        # with ThreadPoolExecutor(4) as executor:
        #     executor.map(get_race_html_by_year_and_mon(year,month),range(1, 13))
    # 今年のデータ
    for year in range(now_datetime.year, now_datetime.year+1):
        for month in range(1, now_datetime.month+1):
                get_race_html_by_year_and_mon(year,month)


def repeat(url):
    list = url.split("/")
    race_id = list[-2]
    save_file_path = save_dir+"/"+race_id+'.html'
    if not os.path.isfile(save_file_path): # まだ取得していなければ取得
        response = requests.get(url)
        response.encoding = response.apparent_encoding # https://qiita.com/nittyan/items/d3f49a7699296a58605b
        html = response.text
        #time.sleep(0.01)
        with open(save_file_path, 'w') as file:
            file.write(html)
def get_race_html_by_year_and_mon(year,month):
    global save_dir
    with open(RACR_URL_DIR+"/"+str(year)+"-"+str(month)+".txt", "r") as f:
        save_dir = RACR_HTML_DIR+"/"+str(year)+"/"+str(month)
        my_makedirs(save_dir)
        urls = f.read().splitlines()

        file_list = os.listdir(save_dir) # get all file names
        
        # 取得すべき数と既に保持している数が違う場合のみ行う
        if len(urls) != len(file_list):
            logger.info("getting htmls ("+str(year) +" "+ str(month) + ")")
            #for url in urls:
            
                
            
            with ThreadPoolExecutor(8) as executor:
                    executor.map(repeat, urls)
                # if not os.path.isfile(save_file_path): # まだ取得していなければ取得
                #     response = requests.get(url)
                #     response.encoding = response.apparent_encoding # https://qiita.com/nittyan/items/d3f49a7699296a58605b
                #     html = response.text
                #     # time.sleep(0.01)
                #     with open(save_file_path, 'w') as file:
                #         file.write(html)
            logging.info("saved " + str(len(urls)) +" htmls ("+str(year) +" "+ str(month) + ")")
        else:
            logging.info("already have " + str(len(urls)) +" htmls ("+str(year) +" "+ str(month) + ")")


if __name__ == '__main__':
    #url 
    formatter = "%(asctime)s [%(levelname)s]\t%(message)s" # フォーマットを定義
    logging.basicConfig(filename='logfile/'+OWN_FILE_NAME+'.logger.log', level=logging.INFO, format=formatter)

    logger.info("start get race html!")
    year,month = 2022,5
    get_race_html_by_year_and_mon(year,month)
