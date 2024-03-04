# group cities by timezone

from dataclasses import dataclass
from itertools import groupby

@dataclass
class City:
    name: str
    timezone: int

data = [
    City('Чикаго', 2),
    City('Новосибирск', 0),
    City('Уй', 0),
    City('Москва', 0),
    City('Пекин', 1),
]

# Предварительно сортируем.
data = sorted(data, key=lambda item: (item.timezone, len(item.name)))

grouped = {}

for key, group_items in groupby(data, key=lambda item: (item.timezone)):
    grouped[key] = list(group_items)

print(grouped)

for key, group_items in grouped.items():
    print('Key: %s' % key)
    for item in group_items: 
        print('Item: %s' % item)