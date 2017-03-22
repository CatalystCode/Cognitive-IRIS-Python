# Microsoft IRIS API: Python SDK
This repo contains the Python SDK for the Microsoft IRIS API, an offering within [Microsoft Cognitive Services](https://www.microsoft.com/cognitive-services), formerly known as Project Oxford.

> NOTE: This is a work-in-progress.

## Installation

```bash
pip install 
```

## Installation from Source Code

```bash
python setup.py install
```

## Run sample to test images and retrain

A Python SDK sample is also provided, before execution,
please install all components listed below.

### Sample Prerequisite

- [Python 2.7](https://www.python.org/downloads/) (only Python 2 supported due
  to limitation of wxPython)

### Sample Execution

```bash
git clone https://github.com/ritazh/Cognitive-IRIS-Python.git
cd Cognitive-IRIS-Python
export TKEY=
export PKEY=
python test.py <directory path with pattern to test images>
e.g. '/Users/testuser/Documents/testimages/{}/'
```


## License
All Microsoft Cognitive Services SDKs and samples are licensed with the MIT License. For more details, see
[LICENSE](</LICENSE.md>).

## Developer Code of Conduct
Developers using Cognitive Services, including this sample, are expected to follow the “Developer Code of Conduct for Microsoft Cognitive Services”, found at [http://go.microsoft.com/fwlink/?LinkId=698895](http://go.microsoft.com/fwlink/?LinkId=698895).
