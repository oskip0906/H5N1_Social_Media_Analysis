import matplotlib.pyplot as plt
import pandas as pd
import os

folder = 'csv_files/outbreaks_by_state'
files = os.listdir(folder)

for file in files:
    
    data = pd.read_csv(f'{folder}/{file}')

    state = file[:-4].split('/')[-1]

    data[['StartWeek', 'EndWeek']] = data['Week'].str.split('/', expand=True)
    data['StartWeek'] = pd.to_datetime(data['StartWeek'])
    data['Cases'] = data['Cases'] / 10**3

    plt.figure(figsize=(10, 6))
    plt.plot(data['StartWeek'], data['Cases'])
    plt.title(f'{state.capitalize()}')
    plt.xlabel('Week')
    plt.ylabel('Cases (Thousands)')
    # plt.show()
    plt.savefig(f'graphs/outbreaks_weekly_by_state/{state}.png')

data = pd.read_csv('csv_files/outbreaks_weekly.csv')

data[['StartWeek', 'EndWeek']] = data['Week'].str.split('/', expand=True)
data['StartWeek'] = pd.to_datetime(data['StartWeek'])
data['Cases'] = data['Cases'] / 10**3

plt.figure(figsize=(10, 6))
plt.plot(data['StartWeek'], data['Cases'])
plt.xlabel('Week')
plt.ylabel('Cases (Thousands)')
# plt.show()
plt.savefig('graphs/oubreaks_weekly_graph.png')