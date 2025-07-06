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
- [ ] Decrease percentage of weights to be scrapepd over time

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




## Low params


MNIST: 
- 82% - 50 - 2/1/10
- 87% - 60 params - 3/1/10
- 87.7% - 20 params, removing all biases and soma_w - 2/10
- 95% - 840 params - 8/8/10



70 params, wiht 4 / 1 / 10 and 30 epochs

fashion mnist 80% accuracy with 110 paramters


    # data config
    dataset = "fashion-mnist"  # "mnist", "fashion-mnist", "cifar10"
    subset_size = None

    # config
    n_epochs = 20 # 15 MNIST, 20 Fashion-MNIST
    lr = 0.003  # 0.07 - SGD
    v_lr = 0.003  # 0.015 - SGD
    b_lr = 0.003  # 0.015 - SGD
    weight_decay = 0.01 #0.001
    batch_size = 256
    n_classes = 10

    if dataset in ["mnist", "fashion-mnist"]:
        in_dim = 28 * 28  # Image dimensions (28x28 MNIST, 32x32x3 CIFAR-10)
    elif dataset == "cifar10":
        in_dim = 32 * 32 * 3
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    # dendriticmodel config
    n_dendrite_inputs = 3
    n_dendrites = 2
    n_neurons = 10
    strategy = "random"  # ["random", "local-receptive-fields", "fully-connected"]

    print("\nRUN NAME: synaptic resampling FALSE\n")

    if dataset in ["mnist", "fashion-mnist"]:
        X_train, y_train, X_test, y_test = load_mnist_data(
            rng, dataset=dataset, subset_size=subset_size
        )
    elif dataset == "cifar10":
        X_train, y_train, X_test, y_test = load_cifar10_data(
            subset_size=subset_size
        )


    print("Preparing model...")
    criterion = CrossEntropy()
    model = Sequential(
        [
            DendriticLayer(
                in_dim,
                n_neurons,
                n_dendrite_inputs=n_dendrite_inputs,
                n_dendrites=n_dendrites,
                strategy=strategy,
                synaptic_resampling=True,
                percentage_resample=0.9,
                steps_to_resample=100,
                scaling_resampling_percentage=False,
                dynamic_steps_size=False,
            ),
            # LeakyReLU(),
            # LinearLayer(n_neurons, n_classes),
        ]