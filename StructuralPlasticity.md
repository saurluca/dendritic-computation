# Structural plasiticy

Deteirme weights to prune:
p ^(1/n) %, n being pruning round

## General ideas

- [x] Incoprotate weight decay?
- [ ] plot weight over picture of a dendrite
- Why does He init work? should I also choose it
- check if dendirte connctions remains coanst with sampling
- TODO: reset update values for Adam for the new grads

### Triggerig resampling

- [x] flat, after n update steps
- [x] flat probability - removed, use dynmaic formula that also chooses weight to resample

### Ideas for Sampling Strategies

- use some kind of function (sigmoid?) to resamble distribution of weights that is ideal. make histogram of current and compare
- [x] flat, prune the smallest p weights
- [x] probality function that assigns higher probaly to smaller weight to be resampled. Very unlikely for big weights, but not impossilbe
- [ ] sample such that the weights convert against mean of He init, or against normal?



## Findings:

The model with sampling can train for a lot longer and still improves, its advantge over the baseline model only increasing. train and test error improve for longer time:


50 epochs MNIST. vannila being without resampling:
train loss Dendritic model 0.0768 vs Vanilla 0.2162
test loss Dendritic model 0.1175 vs Vanilla 0.2338
train accuracy Dendritic model 97.4% vs Vanilla 93.5%
test accuracy Dendritic model 96.4% vs Vanilla 93.0% 

Current implemnation has same training time but better results.