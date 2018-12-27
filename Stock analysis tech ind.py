# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 00:27:49 2018

@author: anushkmore
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 23:39:08 2018

@author: anushkmore
"""

import csv
import re
from bs4 import BeautifulSoup


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

df=[]
file_reader= open('C:\\TAG\\train.csv', "rt")
read = csv.reader(file_reader)


      
      
       
       

      