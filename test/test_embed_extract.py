import torch
from evo2 import Evo2

def main():
    evo2_model = Evo2('evo2_7b')

    sequence = 'ACGT'
    input_ids = torch.tensor(
        evo2_model.tokenizer.tokenize(sequence),
        dtype=torch.int,
    ).unsqueeze(0).to('cuda:0')

    layer_name = 'blocks.28.mlp.l3'

    outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])

    print('Embeddings shape: ', embeddings[layer_name].shape)
    print('Embeddings: ', embeddings)
    print('Outputs shape: ', outputs.shape)
    print('Outputs: ', outputs)

if __name__ == "__main__":
    main() 