import tensorflow as tf
# unused, but forces tensorflow to use bundled keras instead of separate
from tensorflow import keras
from tensorflow.keras import layers

import pulse2percept as p2p
from pulse2percept.models import BiphasicAxonMapModel, BiphasicAxonMapSpatial
from pulse2percept.implants import ArgusII
from collections import OrderedDict

import numpy as np

import pulse2percept as p2p
from pulse2percept.models import BiphasicAxonMapModel, BiphasicAxonMapSpatial # Granley 2021


"""
============================================================================================
                                    Tensorflow Models
============================================================================================
"""

class UniversalBiphasicAxonMapLayer(tf.keras.layers.Layer):
    """ A tensorflow implementation of the model described in (Granley 2021) """
    def __init__(self, p2pmodel, implant, activity_regularizer=None, clip=None, amp_cutoff=True, **kwargs):
        """
        Parameters:
        -----------
        p2pmodel : p2p.models.BiphasicAxonMapModel
            A BiphasicAxonMapModel (Granley 2021) describing the system. Note that rho and axlambda do not need
            to be accurate, this is mainly used to extract axon structure and output grid.
        implant : p2p.implants.ProsthesisSystem
            Any valid p2p implant
        activity_regularizer : keras activity regulizer
        clip : tuple
            (high, low) tuple to clip brightness outputs. Defaults to no clipping
        amp_cutoff : bool
            If true, remove subthreshold amplitudes that will likely not produce a phopshene in
            real subjects
        """
        super(UniversalBiphasicAxonMapLayer, self).__init__(trainable=False,
                                                   name="UniversalBiphasicAxonMap",
                                                   dtype='float32',
                                                   activity_regularizer=activity_regularizer, **kwargs)

        if not (isinstance(p2pmodel, BiphasicAxonMapModel) or isinstance(p2pmodel, BiphasicAxonMapSpatial)):
            raise ValueError("Must pass in a valide BiphasicAxonMapModel")
        if not isinstance(implant, p2p.implants.ProsthesisSystem):
            raise ValueError("Invalid implant")

        # p2pmodel.min_ax_sensitivity = 0.2 don't here
        bundles = p2pmodel.grow_axon_bundles()
        axons = p2pmodel.find_closest_axon(bundles)
        if type(axons) != list:
            axons = [axons]
        axon_contrib = self.calc_axon_sensitivity(p2pmodel, axons, pad=True).astype(np.float32)
        self.axon_contrib = tf.constant(axon_contrib, dtype='float32')

        self.percept_shape = p2pmodel.grid.shape
        self.thresh_percept = tf.constant(p2pmodel.thresh_percept, dtype='float32')

        # Get implant parameters
        self.n_elecs = len(implant.electrodes)
        self.elec_x = tf.constant([implant[e].x for e in implant.electrodes], dtype='float32')
        self.elec_y = tf.constant([implant[e].y for e in implant.electrodes], dtype='float32')

        self.clip = False
        if isinstance(clip, tuple):
            self.clip = True
            self.clipmin = clip[0]
            self.clipmax = clip[1]
        self.amp_cutoff = amp_cutoff

    def compute_output_shape(self, input_shape):
        batched_percept_shape = tuple([input_shape[0]] + list(self.percept_shape))
        return batched_percept_shape



    @tf.function(jit_compile=True)
    def call(self, inputs):
        """ Modified implementation of predict_percept from (Granley 2021) to account for varying patient parameters
        without recompilation

        Parameters:
        ------------
        inputs : list of [stimuli, phis]
            stimuli : tensor with shape (batch, n_elecs, 3)
                array containing freqs, amps, and pdurs for each electrode in the array
            phis : list
                list with 12 patient specific parameters [rho, axlambda, a0-a9]"""

        freq = inputs[0][:, :, 0]
        amp = inputs[0][:, :, 1]
        pdur = inputs[0][:, :, 2]

        rho = inputs[1][:, 0][:, None]
        axlambda = inputs[1][:, 1][:, None]
        a0 = inputs[1][:, 2][:, None]
        a1 = inputs[1][:, 3][:, None]
        a2 = inputs[1][:, 4][:, None]
        a3 = inputs[1][:, 5][:, None]
        a4 = inputs[1][:, 6][:, None]
        a5 = inputs[1][:, 7][:, None]
        a6 = inputs[1][:, 8][:, None]
        a7 = inputs[1][:, 9][:, None]
        a8 = inputs[1][:, 10][:, None]
        a9 = inputs[1][:, 11][:, None]

        scaled_amps = (a1 + a0*pdur) * amp
        # bright
        F_bright = a2 * scaled_amps + a3 * freq
        # force 0 if amp too low
        if self.amp_cutoff:
            F_bright = tf.where(scaled_amps > 0.25, F_bright, tf.zeros_like(F_bright))
        # size
        min_f_size = 10**2 / (rho**2)
        F_size = a5 * scaled_amps + a6
        F_size = tf.maximum(F_size, min_f_size)
        # streak
        min_f_streak = 10**2 / (axlambda ** 2)
        F_streak = a9 - a7 * pdur ** a8
        F_streak = tf.maximum(F_streak, min_f_streak)

        eparams = tf.stack([F_bright, F_size, F_streak], axis=2)

        # apply axon map
        d2_el = (self.axon_contrib[:, :, 0, None] - self.elec_x)**2 + (self.axon_contrib[:, :, 1, None] - self.elec_y)**2
        intensities = (eparams[:, None, None, :, 0] *
                       tf.math.exp((-d2_el[None, :, :, :] / (2. * rho**2 * eparams[:, :, 1])[:, None, None, :]) +
                       (self.axon_contrib[None, :, :, 2, None] / ((axlambda** 2 * eparams[:, :, 2])[:, None, None, :]))))

        intensities = tf.reduce_max(tf.reduce_sum(intensities, axis=-1), axis=-1)
        intensities = tf.where(intensities > self.thresh_percept, intensities, tf.zeros_like(intensities))
        if self.clip:
            intensities = tf.clip_by_value(intensities, self.clipmin, self.clipmax)

        batched_percept_shape = tuple([-1] + list(self.percept_shape))
        intensities = tf.reshape(intensities, batched_percept_shape)

        return intensities


    def calc_axon_sensitivity(self, p2pmodel, bundles, pad=False):
        """ (from pulse2percept)
        Calculate the sensitivity of each axon segment to electrical current

        This function combines the x,y coordinates of each bundle segment with
        a sensitivity value that depends on the distance of the segment to the
        cell body and ``self.axlambda``.

        The number of ``bundles`` must equal the number of points on
        `self.grid``. The function will then assume that the i-th bundle passes
        through the i-th point on the grid. This is used to determine the bundle
        segment that is closest to the i-th point on the grid, and to cut off
        all segments that extend beyond the soma. This effectively transforms
        a *bundle* into an *axon*, where the first axon segment now corresponds
        with the i-th location of the grid.

        After that, each axon segment gets a sensitivity value that depends
        on the distance of the segment to the soma (with decay rate
        ``self.axlambda``). This is typically done during the build process, so
        that the only work left to do during run time is to multiply the
        sensitivity value with the current applied to each segment.

        If pad is True (set when engine is 'jax'), axons are padded to all have
        the same length as the longest axon

        Parameters
        ----------
        bundles : list of Nx2 arrays
            A list of bundles, where every bundle is an Nx2 array consisting of
            the x,y coordinates of each axon segment (retinal coords, microns).
            Note that each bundle will most likely have a different N

        Returns
        -------
        axon_contrib : numpy array with shape (n_points, axon_length, 3)
            An array of axon segments and sensitivity values. Each entry in the
            array is a Nx3 array, where the first two columns contain the retinal
            coordinates of each axon segment (microns), and the third column
            contains the sensitivity of the segment to electrical current.
            The latter depends on ``self.axlambda``. axon_length is set to the
            maximum length of any axon after being trimmed due to min_sensitivity

        """
        xyret = np.column_stack((p2pmodel.grid.xret.ravel(),
                                 p2pmodel.grid.yret.ravel()))
        # Only include axon segments that are < `max_d2` from the soma. These
        # axon segments will have `sensitivity` > `self.min_ax_sensitivity`:
        max_d2 = -2.0 * 3000 ** 2 * np.log(p2pmodel.min_ax_sensitivity)
        axon_contrib = []
        for xy, bundle in zip(xyret, bundles):
            idx = np.argmin((bundle[:, 0] - xy[0]) ** 2 +
                            (bundle[:, 1] - xy[1]) ** 2)
            # Cut off the part of the fiber that goes beyond the soma:
            axon = np.flipud(bundle[0: idx + 1, :])
            # Add the exact location of the soma:
            axon = np.concatenate((xy.reshape((1, -1)), axon), axis=0)
            # For every axon segment, calculate distance from soma by
            # summing up the individual distances between neighboring axon
            # segments (by "walking along the axon"):
            d2 = np.cumsum(np.sqrt(np.diff(axon[:, 0], axis=0) ** 2 +
                                   np.diff(axon[:, 1], axis=0) ** 2)) ** 2
            idx_d2 = d2 < max_d2
            sensitivity = -d2[idx_d2] / 2
            idx_d2 = np.concatenate(([False], idx_d2))
            contrib = np.column_stack((axon[idx_d2, :], sensitivity))
            axon_contrib.append(contrib)

        if pad:
            # pad to length of longest axon
            axon_length = max([len(axon) for axon in axon_contrib])
            axon_sensitivities = np.zeros((len(axon_contrib), axon_length, 3))
            for i, axon in enumerate(axon_contrib):
                original_len = len(axon)
                if original_len >= axon_length:
                    axon_sensitivities[i] = axon[:axon_length]
                elif original_len != 0:
                    axon_sensitivities[i, :original_len] = axon
                    axon_sensitivities[i, original_len:] = axon[-1]

            del axon_contrib
            return axon_sensitivities
        else:
            return axon_contrib



class NaiveEncoding(tf.keras.layers.Layer):
    """ Implements the naive encoding resembling that used in Argus implants"""
    def __init__(self, implant, mode='amp', stimrange=(0, 4), maxval=2, thresh=0.75, freq=20, amp=1, **kwargs):
        """
        Parameters :
        ------------
        implant : p2p.implants.ProsthesisSystem
            Any valid p2p implant
        mode : 'amp' or 'freq'
            Whether to encode amplitude or frequency. Default is amplitude
        stimrange : (low, high)
            Range of amplitudes to map pixel values to
        maxval : float
            The max pixel value possible in input images
        thresh : float
            Mask amplitudes lower than thresh
        freq : float
            Default frequency value to use when mode='amp'
        amp : float
            Default amplitude to use when mode='freq'
        """
        super(NaiveEncoding, self).__init__(trainable=False,
                                                   name="Naive",
                                                   dtype='float32',
                                                   **kwargs)
        self.stimrange = stimrange
        self.maxval = maxval
        self.thresh = thresh
        self.freq = freq
        self.amp = amp
        self.mode = mode

        self.n_elecs = len(implant.electrodes)
        self.array_shape = implant.earray.shape
        self.elec_x = tf.constant([implant[e].x for e in implant.electrodes], dtype='float32')
        self.elec_y = tf.constant([implant[e].y for e in implant.electrodes], dtype='float32')

    def compute_output_shape(self, input_shape):
        batched_percept_shape = tuple([input_shape[0], self.n_elecs, 3])
        return batched_percept_shape

    @tf.function()
    def call(self, inputs):
        """ Modified implementation of predict_percept from (Granley 2021) to account for varying patient parameters
        without recompilation

        Parameters:
        ------------
        inputs : list of [stimuli, phis]
            stimuli : tensor with shape (batch, n_elecs, 3)
                array containing freqs, amps, and pdurs for each electrode in the array
            phis : list
                list with 12 patient specific parameters [rho, axlambda, a0-a9]"""

        targets = inputs
        target_resized = tf.reshape(tf.image.resize(targets, self.array_shape, antialias=True), (tf.shape(targets)[0], -1))
        # target_resized = tf.reshape(rs, [tf.shape(rs)[0], -1])
        target_scaled = target_resized / self.maxval * (self.stimrange[1] - self.stimrange[0]) + self.stimrange[0]
        if self.mode == 'amp':
            amps = tf.where(target_scaled > self.thresh, target_scaled, tf.zeros_like(target_scaled))
            freqs = tf.where(target_scaled > self.thresh, tf.ones_like(target_scaled) * self.freq, tf.zeros_like(target_scaled))
        elif self.mode == 'freq':
            freqs = tf.where(target_scaled > self.thresh, target_scaled, tf.zeros_like(target_scaled))
            amps = tf.where(target_scaled > self.thresh, tf.ones_like(target_scaled) * self.amp, tf.zeros_like(target_scaled))
        pdurs = tf.ones_like(amps) * 0.45
        stim = tf.stack([freqs, amps, pdurs], axis=-1)
        return stim




"""
    ============================================================================================
                                          Implants
    ============================================================================================
"""
class RectangleImplant(p2p.implants.ProsthesisSystem):
    """Convenient wrapper for a rectangular p2p implant"""
    def __init__(self, x=0, y=0, z=0, rot=0, shape=(15, 15), r=150./2, spacing=400., eye='RE', stim=None,
                 preprocess=True, safe_mode=False):
        self.safe_mode = safe_mode
        self.preprocess = preprocess
        self.shape = shape
        names = ('A', '1')
        self.earray = p2p.implants.ElectrodeGrid(self.shape, spacing, x=x, y=y, z=z, r=r,
                                    rot=rot, names=names, etype=p2p.implants.DiskElectrode)
        self.stim = stim

        # Set left/right eye:
        if not isinstance(eye, str):
            raise TypeError("'eye' must be a string, either 'LE' or 'RE'.")
        if eye != 'LE' and eye != 'RE':
            raise ValueError("'eye' must be either 'LE' or 'RE'.")
        self.eye = eye
        # Unfortunately, in the left eye the labeling of columns is reversed...
        if eye == 'LE':
            # TODO: Would be better to have more flexibility in the naming
            # convention. This is a quick-and-dirty fix:
            names = self.earray.electrode_names
            objects = self.earray.electrode_objects
            names = np.array(names).reshape(self.earray.shape)
            # Reverse column names:
            for row in range(self.earray.shape[0]):
                names[row] = names[row][::-1]
            # Build a new ordered dict:
            electrodes = OrderedDict()
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to earray:
            self.earray._electrodes = electrodes
    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape, 'safe_mode': self.safe_mode,
                       'preprocess': self.preprocess})
        return params


"""
============================================================================================
                                        Utils
============================================================================================
"""
def load_model(path, model, implant):
    """ Loads a saved HNA
    This is preferred to tf.keras.models.load_model, because the forward model can be
    jit compiled.

    Parameters:
    -----------
    path : str
        Path to saved model file
    model : p2p.models.BiphasicAxonMapModel
        The model used to train the model, used for axon structure. Rho and lambda do not need to match
    implant : p2p.implants
        The implant used to train the model
    """
    nn1 = tf.keras.models.load_model(path)
    return nn1
    # def clone_fn(layer):
    #     if 'UniversalBiphasicAxonMap' in str(type(layer)):
    #         return UniversalBiphasicAxonMapLayer(model, implant, clip=False, amp_cutoff=True)
    #     return layer.__class__.from_config(layer.get_config())

    # nn = tf.keras.models.clone_model(nn1, clone_function=clone_fn)
    # nn.set_weights(nn1.get_weights())
    # del nn1
    # return nn

def load_mnist(model, scale=2.0, pad=2):
    """ Loads mnist data, rescales brightness, and pads"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255 * scale
    x_train = tf.image.resize_with_pad(x_train.reshape((-1, 28, 28, 1)), model.grid.shape[0]-2*pad, model.grid.shape[1]-2*pad, antialias=True)[:, :, :, 0]
    x_train = np.pad(x_train, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    x_train = x_train.reshape((-1, model.grid.shape[0], model.grid.shape[1], 1))
    x_test = x_test.astype('float32') / 255 * scale
    x_test = tf.image.resize_with_pad(x_test.reshape((-1, 28, 28, 1)), model.grid.shape[0]-2*pad, model.grid.shape[1]-2*pad, antialias=True)[:, :, :, 0]
    x_test = np.pad(x_test, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    x_test = x_test.reshape((-1, model.grid.shape[0], model.grid.shape[1], 1))

    return (x_train, y_train), (x_test, y_test)

def get_vgg_loss(layer=-5, beta=0.00008, beta_mae=1):
    """Joint perceptual loss WITHOUT the laplacian term"""

    vgg = tf.keras.applications.VGG19(include_top=False)
    vgg = tf.keras.Model(vgg.inputs, vgg.layers[layer].output)
    vgg.trainable=False

    def vgg_loss(ytrue, ypred):
        mae = tf.reduce_mean(tf.math.abs(ytrue - ypred), axis=[-1, -2, -3])

        ytrue = tf.expand_dims(ytrue, -1)
        ypred = tf.expand_dims(ypred, -1)
        ytrue = tf.image.resize(tf.image.grayscale_to_rgb(ytrue*255), (224, 224))
        ypred = tf.image.resize(tf.image.grayscale_to_rgb(ypred*255),(224, 224))

        ytrue = tf.keras.applications.vgg19.preprocess_input(ytrue)
        ypred = tf.keras.applications.vgg19.preprocess_input(ypred)

        return beta_mae * mae + beta * tf.reduce_mean((vgg(ytrue) - vgg(ypred))**2, axis=[-1, -2, -3])

    return tf.function(vgg_loss)

def get_patient_params(model, targets, targets_test):
    """Returns patient params (phi) for a patient described by model"""
    model_params = np.tile([model.rho, model.axlambda] +  [getattr(model, 'a'+str(i)) for i in range(0, 10)], (len(targets), 1))
    model_params_test = np.tile([model.rho, model.axlambda] +  [getattr(model, 'a'+str(i)) for i in range(0, 10)], (len(targets_test), 1))

    return model_params, model_params_test
