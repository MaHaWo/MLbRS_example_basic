# MLbRS_example
This is an example experiment repository for the 'Best practices in Machine Learning-based Research Software' course given in February 2026 at  Heidelberg University.

## how to use 
- install [uv](https://docs.astral.sh/uv/)

- Clone this repository to your systme
```bash
git clone git@github.com:MaHaWo/MLbRS_example.git
```
and create a virtual environment:

```bash
uv venv
```

and activate it 
```bash 
source ./.venv/bin/activate 
```

then clone the library. See [here](https://github.com/MaHaWo/MLbRS_example_library) for instructions. 

- go to the root directory of the repository and install the library into your virtual environment. It doesn't have a pip release, so we need to 
install from source 

```bash 
uv pip install ./path/to/MLbRS_example_library 
```

- install the dependencies of the experiment library 

```bash 
uv install 
```

- run the example: 

```python3
python3 ./src/main.py --config ./configs/config_basic.yaml
```

you then should see your model train and get a data/MNIST directory and a results/experiment_YYYYMMDD_HHmmSS directory containing the results and the last model snapshot.

