docker run -it --rm -p 1337:1337 --name tensorflow -u $(id -u):$(id -g) --network host -v ${PWD}:/workdir -w /workdir tensorflow/tensorflow:latest-gpu bash

docker run -u $(id -u):$(id -g)
### Setup

````
conda create --name quickstart
conda activate quickstart
pip install -r requirements.txt
````
