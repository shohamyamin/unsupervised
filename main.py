import sys

sys.path.append('./clustering')


from gas_kmeans import gas_kmeans
from song_clustering import song_clustering


def main():
    gas_kmeans()
    song_clustering()

if __name__ == '__main__':
    main()
