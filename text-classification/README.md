```
docker run -it --rm --gpus all --network host -v ${PWD}:/workdir -w /workdir tensorflow/tensorflow:latest-gpu bash -c "\
pip install -r requirements.txt && bash"
```