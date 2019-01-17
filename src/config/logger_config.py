import yaml


class Logger:

    def __init__(self, filename, ):
        self.filename = filename

    def log(self, environment, timestamp, algorithm, parameters, generations=None, total_steps=None, score=None):
        if generations:
            run = {'run':
                       {'environment': environment,
                        'timestamp': timestamp,
                        'algorithm': algorithm,
                        'parameters': parameters,
                        'n_generation': generations,
                        'score': int(score),
                        }
                   }
        else:
            run = {'run':
                       {'environment': environment,
                        'timestamp': timestamp,
                        'algorithm': algorithm,
                        'parameters': parameters,
                        'total_steps': total_steps,
                        'score': int(score),
                        }

                   }
        with open(self.filename, 'a') as f:
            yaml.safe_dump(run, f)
