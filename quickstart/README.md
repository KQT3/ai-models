```
docker run -it --rm --name tensorflow --network host -p 1337:1337 -u $(id -u):$(id -g) -v ${PWD}:/workdir \
-w /workdir tensorflow/tensorflow:latest-gpu bash
```

### Setup

````
conda create --name quickstart
conda activate quickstart
pip install -r requirements.txt
````
