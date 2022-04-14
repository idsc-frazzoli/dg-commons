import os
import yaml


path = os.path.dirname(os.path.abspath(__file__))
sensing_curves_path = os.path.join(path, 'sensing_performance_curves.yaml')
with open(sensing_curves_path, 'rb') as file:
    data = yaml.full_load(file)
    print(len(data.keys()))

# for key in data.keys():
#     name = key + ".yaml"
#     with open(name, 'w') as file:
#         documents = yaml.dump(data[key], file)
