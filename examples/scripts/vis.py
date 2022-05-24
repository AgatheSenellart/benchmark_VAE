import matplotlib.pyplot as plt

def plot_embeddings_colorbars(emb0,emb_l0,filepath):
    fig, ax = plt.subplots(1,1)
    sc1 = ax.scatter(emb0[:,0],emb0[:,1], c = emb_l0)
    fig.colorbar(sc1,ax=ax)
    plt.savefig(filepath)
    plt.close()