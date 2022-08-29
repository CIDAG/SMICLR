import argparse


def arg_parse():

    parser = argparse.ArgumentParser(
        description='SMICLR Contrastive Framework.'
    )

    parser.add_argument(
        '--target',
        dest='target',
        type=int,
        default=0,
        help='Target property'
    )
    parser.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=101,
        help='Number of epochs'
    )
    parser.add_argument(
        '--no-lr-decay',
        dest='lr_decay',
        action='store_false',
        help='Disable learning decay'
    )
    parser.add_argument(
        '--lr',
        dest='lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        dest='weight_decay',
        type=float,
        default=1e-6,
        help='Weight decay'
    )


    parser.add_argument(
        '--bidirectional',
        dest='bidirectional',
        action='store_true',
        help='Bidirectional RNN layer'
    )
    parser.add_argument(
        '--num-layers',
        dest='num_layers',
        type=int,
        default=1,
        help='Number of RNN layers'
    )


    parser.add_argument(
        '--temperature',
        dest='temperature',
        type=float,
        default=.1,
        help='Temperature'
    )
    parser.add_argument(
        '--batch',
        dest='batch_size',
        type=int,
        default=256,
        help='Batch size'
    )
    parser.add_argument(
        '--sup',
        dest='sup',
        action='store_true',
        help='Supervised training'
    )


    parser.add_argument(
        '--sup-train-size',
        dest='sup_size',
        type=int,
        default=5000,
        help='Size of the supervised set'
    )
    parser.add_argument(
        '--dataset-path',
        dest='dataset_path',
        type=str,
        default='data',
        help='Folder to save the QM9 dataset'
    )


    parser.add_argument(
        '--percent',
        dest='percent',
        type=float, 
        default=.2, 
        help='Augmentation ratio'
    )
    parser.add_argument(
        '--xyz-perb',
        dest='xyz_pertub',
        type=float,
        default=.04,
        help='Maximum noise to add in the XYZ'
    )
    parser.add_argument(
        '--node-drop',
        dest='node_drop',
        action='store_true',
        help='If selected, then use the node drop augmentation'
    )
    parser.add_argument(
        '--subgraph',
        dest='subgraph',
        action='store_true',
        help='If selected, then use the subgraph augmentation'
    )
    parser.add_argument(
        '--edge-perturbation',
        dest='edge_perturbation',
        action='store_true',
        help='If selected, then use the edge perturbation augmentation'
    )
    parser.add_argument(
        '--attribute-masking',
        dest='attribute_masking',
        action='store_true',
        help='If selected, then use the attribute masking augmentation'
    )
    parser.add_argument(
        '--position-masking',
        dest='masking_pos',
        action='store_true',
        help='If selected, then use the XYZ masking augmentation'
    )
    parser.add_argument(
        '--enumeration',
        dest='enumeration',
        action='store_true',
        help='If selected, then use the SMILES enumeration augmentation'
    )
    parser.add_argument(
        '--smiles-noise',
        dest='smiles_noise',
        action='store_true',
        help='If selected, then use the SMILES noise augmentation'
    )


    parser.add_argument(
        '--output',
        dest='output',
        type=str,
        default='exp',
        help='Output folder'
    )
    parser.add_argument(
        '--load-weights',
        dest='load_weights', 
        type=str,
        help='Path to load the weights of the model'
    )

    return parser.parse_args()
