# Ignore a bunch of deprecation warnings
import warnings
import copy
import os
import sys
import time
# from ddsp.colab.colab_utils import audio_bytes_to_np
import crepe
import ddsp
import ddsp.training
import wave
import pyaudio
import soundfile as sf
from ddsp.training.postprocessing import (
    detect_notes, fit_quantile_transform
)
from ddsp.training.plotting import specplot
import gin
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
class FaceDecoder(ddsp.training.models.Autoencoder):
    def __init__(self):
        super().__init__()

    def scale_f0_hz(self, f0_hz):
        F0_RANGE = ddsp.spectral_ops.F0_RANGE
        ret = ddsp.core.hz_to_midi(f0_hz) / F0_RANGE
        return ret.numpy()

    def scale_db(self, db):
        """Scales [-DB_RANGE, 0] to [0, 1]."""
        DB_RANGE = ddsp.spectral_ops.DB_RANGE
        return (db / DB_RANGE) + 1.0



    def cap(self, arr, type):
        F0_LO = 20.0
        F0_HI = 15000.0
        DB_LO = 0.0
        DB_HI = 7.9
        LO = 0.0
        HI = 0.0
        if type == 'ld':
            LO = DB_LO
            HI = DB_HI
        else:
            LO = F0_LO
            HI = F0_HI
        for i in range(len(arr)):
            if arr[i] < LO:
                arr[i] = LO
            if arr[i] > HI:
                arr[i] = HI
        return arr

    def call2(self, f0_hz, loudness_db, training=False):
        features = {'f0_hz': f0_hz, 'loudness_db': loudness_db}

        """Run the core of the network, get predictions and loss."""

        # features = self.encode(features, training=training)
        features['f0_hz'] = self.preprocessor.resample(features['f0_hz'])
        features['loudness_db'] = self.preprocessor.resample(features['loudness_db'])
        features['f0_scaled'] = self.preprocessor.resample(self.scale_f0_hz(features['f0_hz']))
        #features['f0_scaled'] = self.preprocessor.resample(features['f0_hz'])
        features['ld_scaled'] = self.preprocessor.resample(features['loudness_db'])
        features.update(self.decoder(features, training=training))

        # Run through processor group.
        pg_out = self.processor_group(features, return_outputs_dict=True)

        # Parse outputs
        outputs = pg_out['controls']
        outputs['audio_synth'] = pg_out['signal']

        if training:
            self._update_losses_dict(
                self.loss_objs, features['audio'], outputs['audio_synth'])

        return outputs['audio_synth'].numpy()


# Helper Functions
def load_audio(audio_path, sr):
    """
    Args:
        audio_path (str): path to audio file
    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file
    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    audio, sr = librosa.load(audio_path, sr=sr, mono=False)
    return audio

def unit_midi_to_hz(midis):
    midis = 127.0 * midis - 69.0
    midis = midis / 12.0
    midis = 440.0 * 2 ** midis
    return midis

if __name__ == "__main__":
    #model = 'trumpet'  # @param ['violin', 'flute', 'flute2', 'trumpet', 'tenor_Saxophone', 'Upload your own (checkpoint folder as .zip)']

    model = sys.argv[1]
    signal = sys.argv[2]
    output = sys.argv[3]

    warnings.filterwarnings("ignore")
    sample_rate = 16000  # 16000
    sample_width = 4


    x=np.load(signal)
    bs_audio_path = ".\\temp.wav"
    secs=x.shape[1]/10
    wavv=np.zeros(int(x.shape[1])*64)
    sf.write(bs_audio_path,wavv,sample_rate)
    bs_audio = load_audio(bs_audio_path, sample_rate)

    if len(bs_audio.shape) == 1:
        bs_audio = bs_audio[np.newaxis, :]
    print('\nExtracting audio features...')

    # Setup the session.
    ddsp.spectral_ops.reset_crepe()

    # Compute features.
    start_time = time.time()

    audio_features = ddsp.training.metrics.compute_audio_features(bs_audio)
    audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
    print('Audio features took %.1f seconds' % (time.time() - start_time))

    # @title Load a model
    # @markdown Run for ever new audio input
    MODEL = model

    PRETRAINED_DIR = '.\\checkpoints\\' + MODEL
    # Pretrained models.
    model_dir = PRETRAINED_DIR
    gin_file = os.path.join(model_dir, 'operative_config-0.gin')

    # Load the dataset statistics.
    DATASET_STATS = None
    dataset_stats_file = os.path.join(model_dir, 'dataset_statistics.pkl')
    print(f'Loading dataset statistics from {dataset_stats_file}')
    try:
        if tf.io.gfile.exists(dataset_stats_file):
            with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
                DATASET_STATS = pickle.load(f)
    except Exception as err:
        print('Loading dataset statistics from pickle failed: {}.'.format(err))

    # Parse gin config,
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)

    # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
    ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
    ckpt_name = ckpt_files[0].split('.')[0]
    ckpt = os.path.join(model_dir, ckpt_name)

    # Ensure dimensions and sampling rates are equal
    time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
    n_samples_train = gin.query_parameter('Harmonic.n_samples')
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = int(bs_audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size

    # print("===Trained model===")
    # print("Time Steps", time_steps_train)
    # print("Samples", n_samples_train)
    # print("Hop Size", hop_size)
    # print("\n===Resynthesis===")
    # print("Time Steps", time_steps)
    # print("Samples\n", n_samples)

    gin_params = [
        'Harmonic.n_samples = {}'.format(n_samples),
        'FilteredNoise.n_samples = {}'.format(n_samples),
        'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps),
        'oscillator_bank.use_angular_cumsum = True',  # Avoids cumsum accumulation errors.
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)

    # Trim all input vectors to correct lengths
    for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
        audio_features[key] = audio_features[key][:time_steps]
    audio_features['audio'] = audio_features['audio'][:, :n_samples]

    # Set up the model just to predict audio given new conditioning
    model = FaceDecoder()
    model.restore(ckpt)

    # Build model by running a batch through it.
    start_time = time.time()
    _ = model(audio_features, training=False)
    print('Restoring model took %.1f seconds' % (time.time() - start_time))
    # ============================================
    #
    # DDSP PREPARATIONS DONE
    # LOAD SIGNAL
    #
    # ======================================
    x=np.load(signal)
    x[0]=unit_midi_to_hz(x[0])
    # print("MEAN COORDINATES")
    # print(np.mean(x[0]))
    # print(np.mean(x[1]))
    time.sleep(10)
    plt.plot(x[0])
    plt.savefig("f0.png")
    plt.clf()
    plt.plot(x[1])
    plt.savefig("ld.png")
    outio = model.call2(x[0], x[1])
    if len(outio.shape) == 2:
        outio = outio[0]

    sf.write(output, outio, sample_rate)
