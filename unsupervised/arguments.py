import argparse

def arg_parse():

    parser = argparse.ArgumentParser(description='SMICLR Contrastive model.')

    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Learning rate.')

    parser.add_argument('--bidirectional', dest='bidirectional', action='store_true', help='Bidirectional RNN layer.')
    parser.add_argument('--num-layers', dest='num_layers', type=int, default=1)
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3, help='Number of graph convolution layers before each pooling')    

    parser.add_argument('--temperature', dest='temperature', type=float, default=.1)
    parser.add_argument('--batch', dest='batch_size', type=int, default=128)

    parser.add_argument('--dataset-path', dest='dataset_path', type=str, default='data')

    parser.add_argument('--ratio', dest='ratio', type=float, default=.2)

    parser.add_argument('--node-drop', dest='node_drop', action='store_true')
    parser.add_argument('--subgraph', dest='subgraph', action='store_true')
    parser.add_argument('--edge-pertubation', dest='edge_pertubation', action='store_true')
    parser.add_argument('--attribute-masking', dest='attribute_masking', action='store_true')
    parser.add_argument('--smiles-noise', dest='smiles_noise', action='store_true')

    parser.add_argument('--seed', dest='seed', type=int, default=1234)

    return parser.parse_args()
