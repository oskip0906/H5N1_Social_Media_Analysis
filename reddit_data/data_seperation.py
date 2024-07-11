import pandas as pd

comments_df = pd.read_csv("csv_files/comments.csv")

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

states_counts = {
    'minnesota': 0, 'south_dakota': 0, 'california': 0, 'pennsylvania': 0, 'iowa': 0, 'colorado': 0, 'wisconsin': 0, 'michigan': 0, 'ohio': 0, 'texas': 0, 'washington': 0, 'utah': 0, 'maryland': 0, 'kansas': 0
}

# Updated mapping of subreddits to their respective states
subreddits_to_states = {
    **dict.fromkeys(minnesota_subreddits, 'minnesota'),
    **dict.fromkeys(southdakota_subreddits, 'south_dakota'),
    **dict.fromkeys(northern_california_subreddits + southern_california_subreddits + central_california_subreddits, 'california'),
    **dict.fromkeys(pennsylvania_subreddits, 'pennsylvania'),
    **dict.fromkeys(iowa_subreddits, 'iowa'),
    **dict.fromkeys(colorado_subreddits, 'colorado'),
    **dict.fromkeys(wisconsin_subreddits, 'wisconsin'),
    **dict.fromkeys(michigan_subreddits, 'michigan'),
    **dict.fromkeys(ohio_subreddits, 'ohio'),
    **dict.fromkeys(texas_subreddits, 'texas'),
    **dict.fromkeys(washington_subreddits, 'washington'),
    **dict.fromkeys(utah_subreddits, 'utah'),
    **dict.fromkeys(maryland_subreddits, 'maryland'),
    **dict.fromkeys(kansas_subreddits, 'kansas')
}

subreddits = comments_df['Subreddit'].value_counts().to_dict()

states_dfs = {state: pd.DataFrame() for state in set(subreddits_to_states.values())}

for subreddit in subreddits:
    state = subreddits_to_states.get(subreddit)
    if states_dfs[state].empty:
        states_dfs[state] = comments_df[comments_df['Subreddit'] == subreddit]
    else:
        states_dfs[state] = pd.concat([states_dfs[state], comments_df[comments_df['Subreddit'] == subreddit]])

# print(states_dfs)

for state, state_df in states_dfs.items():
    if state_df.shape[0] < 50:
        continue
    state_df.to_csv(f'csv_files/comments_by_state/{state}.csv', index=False)