import os
from imports import *

# basic wrapper w/ save/load capabilities
class BasicNetwork:
    def __init__(self, folder):
        self.network = None
        if folder is not None:
            self.network = tf.keras.models.load_model(self.filename(folder))

    def save(self, folder):
        fname = self.filename(folder)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        self.network.save(fname)

    def filename(self, folder):
        fname = "{}/{}.h5".format(folder, self.name())
        return fname

    def name(self):
        pass

# Artificial Event Variable Network
class AEVNetwork(BasicNetwork):
    def __init__(self, event_dim, bottleneck_dim, hidden_node_counts, activ, folder=None):
        super().__init__(folder)
        if self.network is not None:
            return

        self.network = tf.keras.Sequential(name="event_variable")
        self.network.add(tf.keras.layers.InputLayer(input_shape=(event_dim,)))
        for node_count in hidden_node_counts:
            self.network.add(tf.keras.layers.Dense(node_count, activation=activ))
        self.network.add(tf.keras.layers.Dense(bottleneck_dim, activation=None))

    def name(self):
        return "AEV_network"

# auxiliary classifier network
class AuxCNetwork(BasicNetwork):
    def __init__(self, param_dim, bottleneck_dim, hidden_node_counts, activ, folder=None):
        super().__init__(folder)
        if self.network is not None:
            return

        self.network = tf.keras.Sequential(name="classifier")
        self.network.add(tf.keras.layers.InputLayer(input_shape=(param_dim+bottleneck_dim,)))
        for node_count in hidden_node_counts:
            self.network.add(tf.keras.layers.Dense(node_count, activation=activ))
        self.network.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    def name(self):
        return "AuxC_network"

# trainable composite network
class EVNComposite:
    def __init__(self, event_dim, param_dim, bottleneck_dim, aev_nodes, aev_activ, aux_nodes, aux_activ, mutual_info=False, folder=None):
        self.AEV = AEVNetwork(event_dim, bottleneck_dim, aev_nodes, aev_activ, folder)
        self.AuxC = AuxCNetwork(param_dim, bottleneck_dim, aux_nodes, aux_activ, folder)

        param_shape = (param_dim,)
        event_shape = (event_dim,)
        param_input_tensor = tf.keras.Input(shape=param_shape, name='param_input')
        event_input_tensor = tf.keras.Input(shape=event_shape, name='event_input')

        if mutual_info:
            # skip AEV
            classifier_input_tensor = tf.keras.layers.concatenate([param_input_tensor, event_input_tensor])
        else:
            AEV_output_tensor = self.AEV.network(event_input_tensor)
            classifier_input_tensor = tf.keras.layers.concatenate([param_input_tensor, AEV_output_tensor])
        classifier_output_tensor = self.AuxC.network(classifier_input_tensor)

        self.network = tf.keras.Model(
            inputs = [param_input_tensor, event_input_tensor],
            outputs = classifier_output_tensor
        )

    def save(self, folder):
        self.AEV.save(folder)
        self.AuxC.save(folder)

# a simplified reimplementation of ModelCheckpoint to handle composite models
class CompositeCheckpoint(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model,
        folder,
        monitor = "val_loss",
        mode = "auto",
    ):
        super().__init__()
        self._supports_tf_logs = True
        self.full_model = model
        self.folder = folder
        self.monitor = monitor
        self.epochs_since_last_save = 0
        self.best = None

        if mode not in ["auto", "min", "max"]:
            logging.warning(
                "ModelCheckpoint mode %s is unknown, fallback to auto mode.",
                mode,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            if self.best is None:
                self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            if self.best is None:
                self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                if self.best is None:
                    self.best = -np.Inf
            else:
                self.monitor_op = np.less
                if self.best is None:
                    self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1

        logs = logs or {}
        # Block only when saving interval is reached.
        from keras.utils import tf_utils
        logs = tf_utils.sync_to_numpy_or_python_type(logs)
        self.epochs_since_last_save = 0

        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                "Can save best model only with %s available, "
                "skipping.",
                self.monitor,
            )
        elif self.monitor_op(current, self.best):
            self.best = current
            self.full_model.save(self.folder)
            with open("{}/{}.txt".format(self.folder, "epoch"), 'w') as efile:
                efile.write(str(epoch))
