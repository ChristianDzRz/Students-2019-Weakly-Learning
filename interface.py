# This is a part of the pipeline which provides an interface
# for implementing any of the 3 algorithms.
from data_module import DataBase
from data_processing_module import DataProcessingModule
from rand_input import rand_output
import sed_eval
import yaml


# Interface
class Algorithm:

    def __init__(self, config):
            self.config = config
            self.interfaceConfig = config['interface']

    def read_data(self, mode):
        dpm = DataProcessingModule(self.config)
        data = DataBase(mode, self.config)
        for i in data.csv_index():
            try:
                audio, sample_rate = data.items(i)
                spectogram_mfcc = dpm.mfcc_method(audio, sample_rate)
                spectogram_fb = dpm.filter_bank_method(audio, sample_rate)
            except TypeError:
                print("\n")

        return data.csv_file

    def generate_submission(self):
        rand_output(r"data/metadata/test/test.csv")

    def evaluate(self):
        reference_event_list = sed_eval.io.load_event_list(
                filename="data/metadata/test/test.csv",
                delimiter='\t')
        estimated_event_list = sed_eval.io.load_event_list(
                filename='export_dataframe.csv',
                delimiter='\t')

        event_labels = reference_event_list.unique_event_labels

        # Create metrics classes, define parameters

        event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=event_labels,
            t_collar=self.interfaceConfig['t_collar']
        )

        for filename in reference_event_list.unique_files:
            reference_event_list_for_current_file = reference_event_list.filter(
                filename=filename
            )

            estimated_event_list_for_current_file = estimated_event_list.filter(
                filename=filename
            )

            event_based_metrics.evaluate(
                reference_event_list=reference_event_list_for_current_file,
                estimated_event_list=estimated_event_list_for_current_file
            )

        # Print event-based metrics as reports
        print(event_based_metrics)

    def load_config(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)


# 1st algorithm
class HMM(Algorithm):
    # function for runnning methods from abstract class
    # using given data (from a path);
    #
    # will  return labeled data
    def train(self, data):
        pass


# 2nd algorithm
class CRNN(Algorithm):

    def train(self, data):
        pass


# 3rd algorithm
class GradientLearning(Algorithm):

    def train(self, data):
        pass
