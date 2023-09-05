from imports import *
from data import *
from models import *
from args import *

def check_train_args(args):
    sum_frac = args.frac_train_class0 + args.frac_train_class1 + args.frac_test
    if abs(sum_frac - 1.0) > 0.01:
        raise ValueError("Fractions (train_class0, train_class1, test) should sum to 1, but actually {}".format(sum_frac))
    if args.frac_validation<0 or args.frac_validation>1:
        raise ValueError("Validation fraction must be between 0 and 1 (given {})".format(args.frac_validation))
    return args

# for class 0
def param_gen(N, ranges, rng):
    # Generate parent mass
    if isinstance(ranges,list):
        # multiple ranges
        probs = np.array([r[-1]-r[0] for r in ranges])
        probs = probs/np.sum(probs)
        # generate number of entries in proportion to size of range
        nums = np.rint(N*probs).astype(int)
        leftover = N - np.sum(nums)
        if leftover!=0:
            leftover_abs = abs(leftover)
            leftover_sign = np.sign(leftover)
            # pick a random index for each leftover entry
            leftover_rand = rng.randint(len(ranges), size=leftover_abs)
            leftover_sum = np.array([np.sum(leftover_rand==i) for i in range(len(ranges))])
            nums = nums + leftover_sign*leftover_sum
            assert N - np.sum(nums) == 0
        M0 = np.concatenate([rng.uniform(r[0], r[-1], size=(n, 1)) for r,n in zip(ranges,nums)])
        # randomize
        rng.shuffle(M0)
    else:
        # one range
        M0 = rng.uniform(ranges[0], ranges[-1], size=(N, 1))
    return M0

def balance_params(params, method, rng):
    # sort to split into multiple datasets by theta
    sorter = np.argsort(params,axis=0)
    params = params[sorter].squeeze(axis=1)
    param_uniq = np.unique(params)
    p_min = len(sorter)
    p_max = 0
    # first loop to get min and max
    for p in param_uniq:
        p_sum = np.sum(params==p)
        p_min = min(p_min,p_sum)
        p_max = max(p_max,p_sum)

    # undersampling
    if method=="undersample":
        replace = False
        p_num = p_min
    # oversampling
    elif method=="oversample":
        replace = True
        p_num = p_max

    # perform sampling using np choice
    # return in terms of original indices
    sampler = None
    for p in param_uniq:
        p_mask = np.squeeze(params==p)
        p_args = sorter[p_mask]
        p_sampler = rng.choice(p_args, size=p_num, replace=replace)
        sampler = p_sampler if sampler is None else np.concatenate((sampler, p_sampler))

    return np.squeeze(sampler)

def train():
    sampling_choices = ["undersample","oversample"]

    eparser = EVNParser("train")
    parser = eparser.parser
    parser.add_argument("--frac-train-class0", type=float, default=0.4, help="fraction of training data for class 0")
    parser.add_argument("--frac-train-class1", type=float, default=0.4, help="fraction of training data for class 1")
    parser.add_argument("--frac-test", type=float, default=0.2, help="fraction of training data for test")
    parser.add_argument("--random-seed", type=int, default=10, help="initial value for random seed")
    parser.add_argument("--bottleneck-dim", type=int, default=1, help="bottleneck dimension")
    parser.add_argument("--batch-size", type=int, default=5000, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs for training")
    parser.add_argument("--frac-validation", type=float, default=0.2, help="fraction of training data to use for validation")
    parser.add_argument("--mutual-info", default=False, action="store_true", help="test mutual info of inputs (skip AEV)")
    parser.add_argument("--aev-nodes", type=int, default=[128, 64, 64, 64, 32], nargs='+', help="hidden layer node counts for AEV")
    parser.add_argument("--aux-nodes", type=int, default=[16, 16, 16], nargs='+', help="hidden layer node counts for aux")
    parser.add_argument("--aev-activ", type=str, default="relu", help="activation function for AEV layers")
    parser.add_argument("--aux-activ", type=str, default="relu", help="activation function for aux layers")
    parser.add_argument("--continue-train", type=str, default="", help="folder path to load an existing network and continue training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--sampling", type=str, default=None, choices=sampling_choices, help="sampling to enforce balance in param values")
    parser.add_argument("--best", default=False, action="store_true", help="save best model (rather than last model)")
    parser.add_argument("--preview", default=False, action="store_true", help="show model details w/o training or saving config")
    # workers?
    args = eparser.parse_args(checker=check_train_args, save=lambda x: not x.preview)
    outf_models = args.outf+"/"+args.model_dir

    # first: set random seed
    rng = set_random(args.random_seed)

    # get physics process & data
    process = eparser.get_process(args)
    events = process.get_events()

    # convert to numpy format
    inputs = process.get_inputs(events)
    params = process.get_params(events)

    # sampling if any (before shuffling etc.)
    if args.sampling:
        sampler = balance_params(params, args.sampling, rng)
    else:
        sampler = np.arange(len(params))

    # random shuffling
    shuffler = rng.permutation(sampler)
    inputs = inputs[shuffler]
    params = params[shuffler]

    # training and testing data
    first_event = 0
    N_train_class0 = int(len(inputs)*args.frac_train_class0)
    if len(process.params)>1:
        raise RuntimeError("Class 0 generation for multi-param case not implemented yet")
    class0_params = param_gen(N_train_class0, first(process.params), rng)
    class0_inputs = inputs[first_event:N_train_class0]
    class0_ytarget = np.zeros(shape=(N_train_class0, 1))

    first_event += N_train_class0
    N_train_class1 = int(len(inputs)*args.frac_train_class1)
    class1_params = params[first_event:first_event+N_train_class1]
    class1_inputs = inputs[first_event:first_event+N_train_class1]
    class1_ytarget = np.ones(shape=(N_train_class1, 1))

    first_event += N_train_class1
    permutation = rng.permutation(N_train_class0 + N_train_class1)
    params_train = np.concatenate([class0_params, class1_params])[permutation]
    inputs_train = np.concatenate([class0_inputs, class1_inputs])[permutation]
    ytarget_train = np.concatenate([class0_ytarget, class1_ytarget])[permutation]

    N_test_heatmap = int(len(inputs)*args.frac_test)
    shuffler_test = shuffler[first_event:first_event+N_test_heatmap] # to save test dataset

    # computed automatically by process
    event_dim = process.inputs_dim
    param_dim = process.params_dim
    assert inputs_train.shape[1] == event_dim
    assert params_train.shape[1] == param_dim

    # create network
    # todo: make choice of network configurable
    model = EVNComposite(event_dim, param_dim, args.bottleneck_dim, args.aev_nodes, args.aev_activ, args.aux_nodes, args.aux_activ, args.mutual_info, folder=args.continue_train if len(args.continue_train)>0 else None)
    if args.verbose:
        print("N_train_class0 = {}, N_train_class1 = {}, N_val_class0 = {}, N_val_class1 = {}, N_test = {}".format(
            int(N_train_class0*(1-args.frac_validation)),
            int(N_train_class1*(1-args.frac_validation)),
            int(N_train_class0*(args.frac_validation)),
            int(N_train_class1*(args.frac_validation)),
            N_test_heatmap,
        ))
        model.AEV.network.summary()
        model.AuxC.network.summary()
        model.network.summary()

    # stop here if just previewing
    if args.preview:
        return

    # training
    # todo: make optimizer, loss configurable? or part of network choice
    optim = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.network.compile(optimizer=optim, loss='binary_crossentropy')
    cb_checkpoint = CompositeCheckpoint(
        model=model,
        folder=outf_models,
        monitor="val_loss",
    )
    callbacks = []
    if args.best:
        callbacks.append(cb_checkpoint)
    history = model.network.fit(x=[params_train, inputs_train], y=ytarget_train, batch_size=args.batch_size, epochs=args.epochs, validation_split=args.frac_validation, callbacks=callbacks)

    # save trained model and extra info
    if not args.best:
        model.save(outf_models)
    np.savez("{}/shuffler_test.npz".format(outf_models), [shuffler_test])
    np.savez("{}/loss.npz".format(outf_models), [history.history['loss'],history.history['val_loss']])

if __name__=="__main__":
    train()
