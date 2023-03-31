import preproccess
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import adjusted_rand_score, confusion_matrix 

from scipy.stats import f
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from sklearn.decomposition import PCA

from scipy.stats import f_oneway



def gas_kmeans():
    data = preproccess.get_data()
    datatrain = np.vstack(data)
    print(datatrain.shape)
    indices = np.argsort(datatrain[:, 0])
    datatrain = datatrain[indices]
    xtrain = datatrain[:,1:129]
    ytrain = datatrain[:,0]
    kmeans_init(xtrain, ytrain)
    return datatrain


def kmeans_init(X,ytrain):
    xtrain = X
    # Fit the KMeans model to the data
    # kmeans.fit(X)
    k=7
    kmeans =  KMeans(n_clusters=k)
    kmeans_labels = kmeans.fit_predict(X)
    cluster_centers = kmeans.cluster_centers_
    
    score = adjusted_rand_score(ytrain, kmeans_labels)





    
    cluster_labels = kmeans.labels_

    print(X.shape)


    # anomanly detection
    # Identify potential anomalies
    anomalies = X[kmeans_labels == -1]
     
    print("anomalies",anomalies)

    # pca=PCA(n_components=3)
    # xtrain=pca.fit_transform(xtrain)

    
    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(111, projection='3d')
    # plt.rcParams['legend.fontsize'] = 11   
    # ax.plot(xtrain[0:2564,0], xtrain[0:2564,1], xtrain[0:2564,2], 'o', markersize=2.5, label='Ethanol')
    # ax.plot(xtrain[2565:5490,0], xtrain[2565:5490,1], xtrain[2565:5490,2], 'o', markersize=2.5, label='Ethylene')
    # ax.plot(xtrain[5491:7131,0], xtrain[5491:7131,1], xtrain[5491:7131,2], 'o', markersize=2.5, label='Ammonia')
    # ax.plot(xtrain[7132:9067,0], xtrain[7132:9067,1], xtrain[7132:9067,2], 'o', markersize=2.5, label='Acetaldehyde')
    # ax.plot(xtrain[9068:12076,0], xtrain[9068:12076,1], xtrain[9068:12076,2], 'o', markersize=2.5, label='Acetone')
    # ax.plot(xtrain[12077:13909,0], xtrain[12077:13909,1], xtrain[12077:13909,2], 'o', markersize=2.5, label='Toluene')
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # ax.set_zlabel('PC3')
    # plt.title("Gas Data 3D PCA")
    # ax.legend(loc='upper right')

    # plt.show()

    # pca=PCA(n_components=2)
    # xtrain=pca.fit_transform(X)

    
    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(111)
    # plt.rcParams['legend.fontsize'] = 11   
    # ax.plot(xtrain[0:2564,0], xtrain[0:2564,1], 'o', markersize=2.5, label='Ethanol')
    # ax.plot(xtrain[2565:5490,0], xtrain[2565:5490,1], 'o', markersize=2.5, label='Ethylene')
    # ax.plot(xtrain[5491:7131,0], xtrain[5491:7131,1],  'o', markersize=2.5, label='Ammonia')
    # ax.plot(xtrain[7132:9067,0], xtrain[7132:9067,1],'o', markersize=2.5, label='Acetaldehyde')
    # ax.plot(xtrain[9068:12076,0], xtrain[9068:12076,1], 'o', markersize=2.5, label='Acetone')
    # ax.plot(xtrain[12077:13909,0], xtrain[12077:13909,1],  'o', markersize=2.5, label='Toluene')
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # plt.title("Gas Data 2D PCA")
    # ax.legend(loc='upper right')

    # plt.show()
    

    # # Calculate the F-test statistic and p-value
    # num_clusters = len(set(cluster_labels))
    # num_samples = X.shape[0]
    # ss_between = sum([sum((X[cluster_labels == i, :] - kmeans.cluster_centers_[i])**2) 
    #                 for i in range(num_clusters)])
    # ss_within = sum([sum((X[cluster_labels == i, :] - np.mean(X[cluster_labels == i, :], axis=0))**2) 
    #                 for i in range(num_clusters)])
    # df_between = num_clusters - 1
    # df_within = num_samples - num_clusters
    # ms_between = ss_between / df_between
    # ms_within = ss_within / df_within
    # f_statistic = ms_between / ms_within
    # p_value = 1 - f.cdf(f_statistic, df_between, df_within)
    # print("num_clusters",num_clusters)
    # print("F-statistic: ", f_statistic)
    # print("P-value: ", p_value)
    # # Instantiate the clustering model and visualizer
    # model = KMeans()
    # visualizer = KElbowVisualizer(model, k=(2,30))

    # visualizer.fit(X)        # Fit the data to the visualizer
    # visualizer.show()        # Finalize and render the figure
    # # Get the labels and centroids of the clusters
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(X)

    # visualize the clusters in a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=labels)
    legend = ax.legend(*scatter.legend_elements(),title="Clusters")
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title("Clustering using kmeans with 3D PCA")
    plt.show()

    # # Plot the data points with different colors for each cluster
    # fig, ax = plt.subplots()
    # scatter = ax.scatter(X[:,0], X[:,1], c=labels)


    # # Add a legend with the cluster names
    # legend = ax.legend(*scatter.legend_elements(),
    #                     title="Clusters")

    # # Add the centroids to the plot
    # for i in range(len(centroids)):
    #     ax.scatter(centroids[i][0], centroids[i][1], marker="x", color="black", s=150)
    # plt.title("Gas data clustering kmeans 7 clustering")
    # plt.show()




    # Get the cluster labels and cluster centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Perform ANOVA on the cluster means
    groups = []
    pvalues = []
    for i in range(X.shape[1]):
        for j in range(7):
            group = X[labels==j, i]
            groups.append(group)
        fvalue, pvalue = f_oneway(*groups)
        pvalues.append(pvalue)
        groups = []
    print("p:",pvalues)
    # Plot the p-values
    fig, ax = plt.subplots()
    x = np.arange(len(pvalues))
    ax.bar(x, pvalues)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(128)])
    ax.set_xlabel('Feature')
    ax.set_ylabel('p-value')
    ax.set_title('GAS ANOVA p-values for K-means clustering with K=7')
    plt.show()

    # # Get the cluster labels and cluster centers
    # labels = kmeans.labels_
    # centers = kmeans.cluster_centers_

    # # Perform ANOVA on the cluster means
    # groups = []
    # for i in range(7):
    #     groups.append(X[labels==i, :])
    # fvalue, pvalue = f_oneway(*groups)

    # # Print the results
    # print("Cluster means:", centers)
    # print("F-value:", fvalue)
    # print("p-value:", pvalue)
    # print("shape F-value:", fvalue.shape)
    # print("shape p-value:", pvalue.shape)

