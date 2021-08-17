# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 2021

@author: Denise O'Sullivan

This script creates a function which will webscrape the worldometers website for population data on countries around the world

Data Source: https://www.worldometers.info/world-population/population-by-country/
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

def webscraper_population_data():
    
    url = 'https://www.worldometers.info/world-population/population-by-country'
    response = requests.get(url)
    
    try:
        response.status_code == 200    
    except requests.exceptions.HTTPError as e:
        # an error occurred if response was not 200
        return "Error: " + str(e)
    
    soup_content = BeautifulSoup(response.content, features='lxml')   
    
    table = soup_content.find('table', id='example2') # TODO: make sure the the table with this id is always present
    
    data_frame = pd.read_html(str(table))[0]
    
    return data_frame
