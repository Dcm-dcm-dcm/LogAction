# LogAction
**LogAction: Consistent Cross-system Anomaly Detection through Logs via Active Domain Adaptation.**
## Requirements
```yaml
nltk==3.8.1
numpy==1.26.2
pandas==2.1.4
PyYAML==6.0.1
scikit_learn==1.3.2
torch==2.1.1
tqdm==4.66.1
transformers==4.36.0
```
## Log data
BGL, Thunderbird, and Zookeeper data are all from the [loghub](https://github.com/logpai/loghub). If you are interested in the datasets, please follow the link to submit your access request.

# Experiment
When initializing the runtime, it's necessary to download and parse the log dataset. Please set the following configuration file in the ./config directory to True.
```yaml
global:
  need_encoding: True
  need_preprocess: True
```
For datasets BGL, Thunderbird, and Zookeeper, their corresponding names in the ./config are bgl, thu, and zoo respectively. To conduct the experiment from BGL to Thunderbird, please execute the following code:
```shell
python3 -u main.py --config bgl_to_thu.yaml
```
If you want to run $LogAction_{wt}$, please set the parameters:
```yaml
global:
  need_encoding: True
  use_transfer_learning: False
```
If you want to run $LogAction_{wa}$, please set the parameters:
```yaml

anomaly_detection:
  active_learning:
    random: True
```
