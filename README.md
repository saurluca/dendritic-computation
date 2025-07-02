# Project

Explore dendritic trees as computational units beyond passive integration.
Implement sparse "dendrite networks" where hidden units represent dendrites connected to
specific spatial patches or random subsets of inputs, testing how this structural prior affects
learning.

## Sources
This work is based on:

1. London, M., & Häusser, M. (2005). Dendritic computation. Annual Review of
Neuroscience, 28, 503-532.

2. Chavlis, S. and Poirazi, P., 2025. Dendrites endow artificial neural networks with
accurate, robust and parameter-efficient learning. Nature Communications, 16(1), p.943.



## Ideas:

- [x] Improve weight initilisation
- [x] shuffle training data for each epoch
- [ ] implement the other 2 dendric input strategies
- [x] implement multiple layers
- [x] batch processing
- [x] gpu viable (jax, cupy?)
- [x] baseline comparison model
- [x] parameter count 
- [x] use leaky relu
- [x] implement Adam
- [ ] weight decay
- [ ] optional: implement dataloader class
- [x] Combine both SGD classes
- [ ] speed up sparse matrix calcuation using CSR
- [ ] add Fashion MNIST


## Findings

LeakyReLU is massivly important for dendritic model to outperform vanilla model.

Larger training data is more important. Vanilla has better results with fewer data (10^2 pictures) but with 10 ^3 dendritic outperforms vannilla, given LeakyRelu, even with fewer parameters. (32 dendrites, 16 input per dendrite)

Vanilla model requires lower learning rate, coverges slower

Vanilla model looses a good amount of performance if the train batches are not shuffled every epoch. Higher train / test error and lower accuracy (0.92 vs 0.83). Dendrite model does not suffer from same problems. While dendritic model largly stays the same.


## Explore

- Effect of momentum on both
- Compare training time of both
- Adam vs SGD for this task
- Compare different number of layers, breadth vs dept