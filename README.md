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
- [x] implement multiple layers
- [x] batch processing
- [x] gpu viable (jax, cupy?)
- [x] baseline comparison model
- [x] parameter count 
- [x] use leaky relu
- [x] implement Adam
- [x] Combine both SGD classes
- [x] add Fashion MNIST
- [ ] implement the other 2 dendric input strategies
- [x] weight decay

**Optional:**
- [ ] speed up sparse matrix calcuation using CSR
- [ ] optional: implement dataloader class


## Findings

LeakyReLU is massivly important for dendritic model to outperform vanilla model.

Larger training data is more important. Vanilla has better results with fewer data (10^2 pictures) but with 10 ^3 dendritic outperforms vannilla, given LeakyRelu, even with fewer parameters. (32 dendrites, 16 input per dendrite)

Vanilla model requires lower learning rate, coverges slower

Vanilla model looses a good amount of performance if the train batches are not shuffled every epoch. Higher train / test error and lower accuracy (0.92 vs 0.83). Dendrite model does not suffer from same problems. While dendritic model largly stays the same.

Local receptive fields requrie more n_dendrites (16 problem, 32 good) to work. otherwise they get out performed by the random strategy and by the vanilla model.

Dendritic model sometimes has considerable time to setup dendrtic inputs

Using cupy Compressed Sparse Row (CSR) matrix represnation, able to achieve around a 50% speedup. Still Dendritic model a lot slower to train then vanilla model. 

## Explore

- Effect of momentum on both
- Compare training time of both
- Adam vs SGD for this task
- Compare different number of layers, breadth vs dept
- Weight decay



## Notes

For comparison: use 3 dendrite layers


## Structural plasiticy

Deteirme weights to prune:
p ^(1/n) %, n being pruning round