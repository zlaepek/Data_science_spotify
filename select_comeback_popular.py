import pandas as pd

# 1.1. Load data
original_data = pd.read_csv('tracks.csv')


def is_comeback(name, artists):
    is_name_equals = original_data['name'] == name
    is_artists_equals = original_data['artists'] == artists
    comeback_df = original_data[is_name_equals & is_artists_equals]
    return comeback_df


""" remake """
comeback_dfs = is_comeback('We Are Young (feat. Janelle Monáe)', '[\'fun.\', \'Janelle Monáe\']').head(1)
comeback_dfs = comeback_dfs.append(is_comeback('Cheerleader - Felix Jaehn Remix Radio Edit', '[\'OMI\']').head(1))

""" season (Christmas) """
comeback_dfs = comeback_dfs.append(is_comeback('All I Want for Christmas Is You', '[\'Mariah Carey\']').head(1))

""" tictok """
# ref : https://www.altpress.com/features/old-alternative-songs-popular-on-tiktok/
comeback_dfs = comeback_dfs.append(is_comeback('All Star', '[\'Smash Mouth\']').head(1))
comeback_dfs = comeback_dfs.append(is_comeback('Photograph', '[\'Nickelback\']').head(1))
comeback_dfs = comeback_dfs.append(is_comeback('Where Is The Love?', '[\'Black Eyed Peas\']').head(1))
comeback_dfs = comeback_dfs.append(is_comeback('Hollaback Girl', '[\'Gwen Stefani\']').head(1))


# print data
print(comeback_dfs['name'])

comeback_dfs.to_csv('comeback_popular.csv', sep=',', na_rep='NaN')
