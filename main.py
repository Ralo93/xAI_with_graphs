from src.helpers.datasets import *



def main():
    
    cora_dataset = CoraDataset('data/cora.npz')
    print(cora_dataset)


    nodes = cora_dataset.node_features
    labels = cora_dataset.labels
    edges = cora_dataset.edges

    print(labels)




    pass





# The standard Python entry point
if __name__ == "__main__":
    main()
