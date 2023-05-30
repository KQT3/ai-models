```
docker run -it --rm --gpus all --network host -v ${PWD}:/workdir -w /workdir tensorflow/tensorflow:latest-gpu bash -c "\
pip install -r requirements.txt && bash"
```

docker run --gpus 1 -it -v ${PWD}:/workdir -w /workdir tensorflow/tensorflow:latest-gpu bash

docker run --gpus all -it -v ${PWD}:/workdir -w /workdir tensorflow/tensorflow:latest-gpu bash
docker run --gpus all -it -v ${PWD}:/workdir -w /workdir tensorflow/tensorflow:latest-gpu bash

d