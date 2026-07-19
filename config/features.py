iqr_features = ['speechiness', 'acousticness', 'liveness', 'tempo', 'track_age']

zscore_features = ['danceability','energy', 'loudness', 'valence', 'duration_sec']

transformations_dict = {
    'yj_transform' : ['danceability', 'loudness'],
    'expo_transform' : ['energy'],
    'log_transform' :['speechiness', 'liveness', 'track_age'],
    'sqrt_transform' : ['acousticness']
}


