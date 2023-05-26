from interface import HMM, CRNN, GradientLearning
import yaml

with open("config.yaml", 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)
    mainConfig = config['main']

model_factory = {'HMM': HMM(config),
                 'CRNN': CRNN(config),
                 'GradientLearning': GradientLearning(config)
                 }

if __name__ == '__main__':

    algorithm = model_factory[mainConfig['model_name']]
    ops = {'generate_submission': 'generate_submission()',
           'evaluate': 'evaluate()',
           'read_data': "read_data('test')",
           'load_config': 'load_config("config.yaml")'
           }

    operations = mainConfig['ops']
    for n in operations:
        if n in ops:
            exec('algorithm.' + ops[n])
