import time
import argparse

import torch

from cuml.cluster import KMeans

# from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import normalized_mutual_info_score


def main(args):
    # Load embeddings
    embeddings_file = torch.load(args.embeddings_file)
    files = list(embeddings_file.keys())
    labels = [file.split('/')[-3] for file in files]
    embeddings = torch.cat(list(embeddings_file.values())).numpy()
    print(f"Embedding shape: {embeddings.shape}")

    # K-Means
    print("KMeans...")
    kmeans_start_time = time.time()
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        random_state=0,
        max_samples_per_batch=1000000,
        verbose=True
    ).fit(embeddings)
    pseudo_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print(f"K-Means duration: {(time.time() - kmeans_start_time)/60:.2f} min")

    # AHC
    if args.n_clusters_ahc > 0:
        print("AHC...")
        ahc_start_time = time.time()
        ahc_labels = AgglomerativeClustering(
            n_clusters=args.n_clusters_ahc
        ).fit_predict(centroids)
        pseudo_labels = [ahc_labels[pl] for pl in pseudo_labels]
        print(f"AHC duration: {(time.time() - ahc_start_time)/60:.2f} min")

    # Print NMI
    nmi_score = normalized_mutual_info_score(labels, pseudo_labels)
    print(f"NMI: {nmi_score}")

    # Export pseudo labels
    with open(args.output_file, 'w') as f:
        for file, pseudo_label in zip(files, pseudo_labels):
            f.write(f"{pseudo_label} {file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'embeddings_file',
        help='Path to embeddings file (.pt).'
    )
    parser.add_argument(
        'output_file',
        help='Path to output file (.txt).'
    )
    parser.add_argument(
        '--n_clusters',
        help='Number of clusters for KMeans.',
        type=int,
        default=50000
    )
    parser.add_argument(
        '--n_clusters_ahc',
        help='Number of clusters for Agglomerative Clustering.',
        type=int,
        default=7500
    )
    args = parser.parse_args()

    main(args)
