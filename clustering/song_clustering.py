import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
import hdbscan
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import adjusted_rand_score, confusion_matrix 

from sklearn.manifold import TSNE

from scipy.stats import f
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from sklearn.decomposition import PCA

from scipy.stats import f_oneway

def song_clustering():
    data_folder = './datasets/fma'


    # import datasets from the 
    spotify_data = pd.read_csv(f'{data_folder}/fma_metadata/echonest.csv', index_col=0, header=[0, 1])
    tracks = pd.read_csv(f'{data_folder}/fma_metadata/tracks.csv', index_col=0, header=[0, 1])
    genres = pd.read_csv(f'{data_folder}/fma_metadata/genres.csv')

    # echonest.csv
    # print(spotify_data.shape)
    # print(spotify_data.head())
    # # echonest.csv

    # print("rrr",spotify_data.columns)
    # print(spotify_data.index)
    # spotify_data.columns = spotify_data.columns.droplevel() 
    # # spotify_data = spotify_data[[('echonest', 'audio_features'),
    # #             "('echonest', 'audio_features').1",
    # #             "('echonest', 'audio_features').2",
    # #             "('echonest', 'audio_features').3",
    # #             "('echonest', 'audio_features').4",
    # #             "('echonest', 'audio_features').5",
    # #             "('echonest', 'audio_features').6",
    # #             "('echonest', 'audio_features').7"]]

    spotify_data = spotify_data.iloc[:, :8]
    # print("fff",spotify_data.columns)

    spotify_data.rename(columns=spotify_data.iloc[0], inplace=True)
    spotify_data.drop(spotify_data.index[0], inplace=True)
    spotify_data.drop(spotify_data.index[0], inplace=True)
    # spotify_data = spotify_data.rename(index={np.nan: 'track_id'})
    spotify_data.index.name = 'track_id'


    # preproccess the data
    tracks = tracks[[('album','date_created'), ('album','date_released'), ('album', 'title'), ('album', 'id'),
                    ('artist', 'id'), ('artist', 'name'),  
                    ('track', 'genre_top'), ('track', 'genres_all'),('track', 'title'), ('track', 'duration')]]
    # tracks.csv
    tracks.columns = tracks.columns.droplevel()
    tracks.columns = ['date_created', 'date_released', 'album', 'album_id', 'artist_id', 'artist', 'genres_top', 'genres_all', 'track', 'duration']

    # print(tracks.describe)
    # tracks preproccess
    # print(tracks.isna().sum())
    # date_released
    tracks['date_released'].fillna(tracks['date_created'], inplace=True)
    # # fill nulls
    tracks['date_released'].fillna(0, inplace=True) 
    tracks.drop('date_created', axis=1, inplace=True)   
    tracks = tracks[tracks['genres_all'] != '[]']   
    tracks['genres_all'] = [map(int, i.strip('][').split(',')) for i in tracks['genres_all']] 
    genres_null = tracks.loc[(tracks['genres_top'].isnull()), ['genres_top', 'genres_all']].copy(deep=True) 
    genres_top = genres[genres.top_level == genres.genre_id]  
    genre_top_dict = dict(zip(genres_top.genre_id, genres_top.title)) 

    for index, row in genres_null.iterrows():                                           # fill null values in genres_top with top genre in genres_all
        for genre in row['genres_all']:
            if genre in genre_top_dict:
                tracks.loc[index, 'genres_top'] = genre_top_dict[genre]
                break
            
    genres_all_dict = dict(zip(genres.genre_id, genres.title))
    tracks['genres_all'] = [[genres_all_dict[j] for j in i] for i in tracks['genres_all']]

    tracks['album'].fillna(tracks['artist'] + '-' + [','.join(i) for i in tracks['genres_all']], inplace=True)     # fill missing album title with corresponding value from artist & genres_all
    tracks['track'].fillna(tracks['artist'] + '-' + tracks['genres_top'], inplace=True)


    # fix types
    tracks['date_released'] = pd.to_datetime(tracks['date_released'], format='%Y-%m-%d %H:%M:%S')
    tracks[['album', 'artist', 'track', 'genres_top']] = tracks[['album', 'artist', 'track', 'genres_top']].astype('string')

    genres['title'] = genres['title'].astype('string')
    spotify_data = spotify_data.apply(pd.to_numeric)


    tracks = tracks.loc[tracks.index.isin(spotify_data.index.values)] 
    spotify_tracks = spotify_data.join(tracks, how='inner') 

    # standardization
    audio_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']
    scaler = StandardScaler()
    spotify_tracks[audio_features] = scaler.fit_transform(spotify_tracks[audio_features])

    print("gener",genres)

    # model development
    # K-Means
    x = spotify_tracks[audio_features]
    k=7
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(x)
    cluster_centers = kmeans.cluster_centers_

    # clustering using PCA
    labels = kmeans.labels_
    # pca = PCA(n_components=3)
    # reduced_data = pca.fit_transform(x)

    # # visualize the clusters in a 3D scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=labels)
    # legend = ax.legend(*scatter.legend_elements(),title="Clusters")
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # ax.set_zlabel('PC3')
    # plt.title("Song Clustering using kmeans with 3D PCA")
    # plt.show()


    # # model development
    # sse = []
    # for k in range(1, 30):
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     kmeans.fit(x)
    #     sse.append(kmeans.inertia_)
    # plt.xlabel("K (number of clusters)")
    # plt.ylabel("SSE (sum of square error)")
    # plt.title('Song Elbow Method')
    # plt.plot(range(1, 30), sse, marker="*")

    # plt.show()


    x = spotify_tracks[audio_features]
    kmeans = KMeans(n_clusters=k, random_state=0)
    y_predicted = kmeans.fit_predict(x)
    spotify_tracks['cluster'] = y_predicted


    pca = PCA(n_components=2)            
    x = pca.fit_transform(x)
    comp_x, comp_y = zip(*x)
    spotify_tracks['x'] = comp_x
    spotify_tracks['y'] = comp_y
    sns.lmplot(data=spotify_tracks, x='x', y='y', hue='cluster', fit_reg=False, height=8, palette='Set1', scatter_kws={'alpha':0.4, 's':30})
    plt.title("Song PCA Clustering by KMeans (k=7)")
    plt.show()


    x = spotify_tracks[audio_features]
    # Define the colors for each cluster
    cluster_colors = {
        0: "red",
        1: "blue",
        2: "green",
        3: "purple",
        4: "orange",
        5: "brown",
        6: "pink"
    }

    # Fit a k-means model to your data
    kmeans = KMeans(n_clusters=7, random_state=0)
    kmeans.fit(x)

    # Obtain cluster assignments for each data point
    cluster_assignments = kmeans.predict(x)

    # Apply t-SNE to the data
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(x)

    # Plot the t-SNE results with each data point colored by its corresponding cluster assignment
    colors = [cluster_colors[c] for c in cluster_assignments]
    for i, color in cluster_colors.items():
        plt.scatter(tsne_results[cluster_assignments == i, 0], tsne_results[cluster_assignments == i, 1], c=color, label=f"Cluster {i}")
    plt.legend()
    plt.title("Song TSNE Clustering by KMeans (k=7)")
    plt.show()

    x = spotify_tracks[audio_features]
    # Get the cluster labels and cluster centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_




    # create HDBSCAN model
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)

    # fit the model to the data
    clusterer.fit(x)

    # get the labels assigned by the model
    labels = clusterer.labels_

    # get the number of clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Number of clusters:", n_clusters)


    # apply PCA to project the data onto 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x)

    # plot the data points with different colors for ground truth and predicted labels

    # true_lables = [ genres['gener_id'][genres['#tracks'][col == track_index].index] for track_index in x[:][0]]
    print(genres.columns)
    print(x.iloc[:, 0])
    data = x.iloc[:, 0]
    print("dcd",data)
    true_labels = []
    for track_index in x.iloc[:, 0]:
        true_labels.append(genres.loc[genres["#tracks"] == track_index,'genre_id'])
    # true_lables = [genres.loc[genres["#tracks"] == track_index,'gener_id'] for track_index in x[:][0]]
    print(true_labels[0:10])
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(X_pca[:,0], X_pca[:,1], c=true_labels, cmap='rainbow')
    ax[0].set_title("Ground Truth Labels")
    ax[1].scatter(X_pca[:,0], X_pca[:,1], c=pred_labels, cmap='rainbow')
    ax[1].set_title("Predicted Labels")

    # compute ARI score
    ari_score = adjusted_rand_score(true_labels, pred_labels)
    print("Adjusted Rand Index:", ari_score)

    plt.show()

