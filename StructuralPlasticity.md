# Structural plasiticy

Deteirme weights to prune:
p ^(1/n) %, n being pruning round

## General ideas

- [x] Incoprotate weight decay?
- [x] plot weight over picture of a dendrite
- [x] Why does He init work? should I also choose it -> only good for weight init
- [x] number check if dendirte connctions remains coanst with sampling -> yes
- [x] reset update values for Adam for the new grads -> worsens results if reset
- [x] implement local recpeitve fields, then resample based on gaussian distribution -> same to worse results
- [ ] run param search over night.
- [ ] Decrease frequency of param removal for later batches
- [ ] rensure during the last n batches of training, before eval, not to remove batches
- [ ] mean or rolling average of a weight sntead of current value
- [ ] calculate entropy (noiseis) of weight distributon for a dendrite, once at beginning and at the end to compare

### Triggerig resampling

- [x] flat, after n update steps
- [x] flat probability - removed, use dynmaic formula that also chooses weight to resample

### Ideas for Sampling Strategies

- [ ] use some kind of function (sigmoid?) to resamble distribution of weights that is ideal. make histogram of current and compare
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

- lrf_resampling is worse then random resampling
- current implementation of local recptive fields similar or worse to random init with and without sampling