import requests
import pandas as pd
from datetime import datetime

link_structure = 'https://api.pullpush.io/reddit/search/comment/?'

param_structure = {
    'sort': 'asc',
    'size': 100,
    'after': int(datetime.strptime('2022-02-08', "%Y-%m-%d").timestamp())
}

parameters = [
    {
        'q': '"bird%20flu"',
        'param_structure': param_structure
    },
    {
        'q': '"avian%20flu"',
        'param_structure': param_structure
    },
    {
        'q': '"avian%20influenza"',
        'param_structure': param_structure
    },
    {
        'q': 'h5n1',
        'param_structure': param_structure
    },
]

comments_links = []

for parameter in parameters:
    param_structure = parameter['param_structure']
    param_query = parameter['q']
    comments_links.append(f'{link_structure}q={param_query}&{"&".join(f"{key}={value}" for key, value in param_structure.items())}')

# print(comments_links)

minnesota_subreddits = ['minnesota', 'Minneapolis', 'StPaul', 'Duluth', 'RochesterMN', 'BloomingtonMN', 'Burnsville', 'Eagan', 'MapleGrove', 'Mankato']
southdakota_subreddits = ['SouthDakota', 'SiouxFalls', 'RapidCity', 'BrookingsSD', 'MitchellSD', 'WatertownSD', 'AberdeenSD', 'VermillionSD', 'Spearfish', 'HuronSD']
northern_california_subreddits = ['norcal', 'SanFrancisco', 'Oakland', 'Sacramento', 'Berkeley', 'Marin', 'Napa', 'Humboldt', 'SantaRosa', 'Eureka']
southern_california_subreddits = ['socal', 'LosAngeles', 'SanDiego', 'OrangeCounty', 'LongBeach', 'Irvine', 'SantaBarbara', 'PalmSprings', 'Riverside', 'Ventura']
central_california_subreddits = ['CentralValley', 'Fresno', 'Bakersfield', 'Modesto', 'Visalia', 'Stockton', 'Merced', 'SantaCruz', 'Monterey', 'Salinas']
pennsylvania_subreddits = ['Pennsylvania', 'philadelphia', 'pittsburgh', 'Harrisburg', 'Allentown', 'Erie', 'Scranton', 'ReadingPA', 'LancasterPA', 'StateCollege']
iowa_subreddits = ['Iowa', 'desmoines', 'cedarrapids', 'IowaCity', 'Ames', 'Dubuque', 'Waterloo', 'SiouxCity', 'Ankeny', 'CouncilBluffs']
colorado_subreddits = ['Colorado', 'Denver', 'ColoradoSprings', 'Boulder', 'FortCollins', 'Greeley', 'AuroraCO', 'GoldenCO', 'LovelandCO', 'Pueblo']
wisconsin_subreddits = ['wisconsin', 'milwaukee', 'madisonwi', 'GreenBay', 'Kenosha', 'Racine', 'Appleton', 'Waukesha', 'Oshkosh', 'LaCrosse']
michigan_subreddits = ['Michigan', 'Detroit', 'grandrapids', 'AnnArbor', 'Lansing', 'Kalamazoo', 'Flint', 'Muskegon', 'EastLansing', 'TraverseCity']
ohio_subreddits = ['Ohio', 'Columbus', 'Cleveland', 'cincinnati', 'Akron', 'Toledo', 'Dayton', 'Youngstown', 'Canton', 'Mansfield']
texas_subreddits = ['texas', 'austin', 'Dallas', 'houston', 'sanantonio', 'FortWorth', 'ElPaso', 'Plano', 'Arlington', 'Lubbock']
washington_subreddits = ['Washington', 'Seattle', 'Spokane', 'Tacoma', 'Bellevue', 'Everett', 'KentWA', 'Yakima', 'Renton', 'Bellingham']
utah_subreddits = ['Utah', 'saltlakecity', 'Ogden', 'Provo', 'Logan', 'StGeorge', 'Sandy', 'Murray', 'Orem', 'Lehi']
maryland_subreddits = ['Maryland', 'baltimore', 'SilverSpring', 'Annapolis', 'FrederickMD', 'ColumbiaMD', 'Rockville', 'Bethesda', 'Towson', 'Germantown']
kansas_subreddits = ['Kansas', 'Wichita', 'OverlandPark', 'KansasCityKS', 'Topeka', 'Olathe', 'Lawrence', 'Shawnee', 'ManhattanKS', 'Lenexa']

all_subreddits = minnesota_subreddits + southdakota_subreddits + northern_california_subreddits + southern_california_subreddits + central_california_subreddits + \
                pennsylvania_subreddits + iowa_subreddits + colorado_subreddits + wisconsin_subreddits + michigan_subreddits + ohio_subreddits + texas_subreddits + \
                washington_subreddits + utah_subreddits + maryland_subreddits + kansas_subreddits

links = []

for subreddit in all_subreddits:
    for link in comments_links:
        links.append(f'{link}&subreddit={subreddit}')

# print(links)

comments_data = []

for link in links:
    comment_response = requests.get(link)
    comments = comment_response.json()
    count = 0
    for comment in comments['data']:
        try:
            date = datetime.fromtimestamp(comment['created_utc']).strftime('%Y-%m-%d %H:%M:%S')
            subreddit = link.split('&')[-1].split('=')[-1]
            comments_data.append({'Comment': comment['body'], 'Date': date, 'Subreddit': subreddit})
            count += 1
        except Exception as e:
            print(f"error processing comment")
            continue
    print(count)

df = pd.DataFrame(comments_data)

print(df)

new_df = df.drop_duplicates(subset=['Comment'])
new_df.to_csv('csv_files/comments.csv', index=False)