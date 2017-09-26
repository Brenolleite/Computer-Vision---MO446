# Verificador de notas automatico

import numpy as np
import requests
from bs4 import BeautifulSoup

RA = <COLOQUE SEU RA AQUI>
n = total = 0

page = requests.get("https://docs.google.com/spreadsheets/d/1MHTy1PezeyXD2c8aUMNtei7QzS0A8JrFJThsSMslLdc/pubhtml?gid=228104625&single=true")
soup = BeautifulSoup(page.content, 'html.parser')
for nota in soup.find(string=RA).parent.parent.find_all('td', class_="s6")[1:]:
    n += 1
    total += float(nota.text)

print("Media = {0}".format(round(total/n, 2)))