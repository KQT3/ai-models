```
docker run -it --rm -p 1337:1337 --name tensorflow -u $(id -u):$(id -g) --network host -v ${PWD}:/workdir -w /workdir \
tensorflow/tensorflow:latest-gpu bash
```

### Setup

````
conda create --name quickstart
conda activate quickstart
pip install -r requirements.txt
````
