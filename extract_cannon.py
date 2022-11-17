import csv
import numpy as np
import pandas
import pickle
from langdetect import detect
import json

with open('data/CONAN/CONAN.json', 'r', encoding= 'utf-8') as f:
  data = json.load(f)

connan = data['conan']

with open('CONAN_for_VAE.txt', 'w', encoding= 'utf-8') as f:
    for item in connan:
        counter = item['counterSpeech']
        if detect(counter) == 'en':
            f.write(item['counterSpeech'] + '\n')



