# Perceptron

- Sample command line
```
./perceptron.py --noDev --iterations 1
```

- Arguments
    - If `--noDev` is not provided, then we experiment the performance on development data as a function of iterations. A plot will show up.
    - If `--iterations` is not provided, then the default iteration number is 10.

- Development data
    - When using this data, after each iteration of training, we test the performance on development data and get the performance.
    - When the training stops, we pick the weight vector with which the performance on development data is the best.
    - Experiment plot, accuracy as a function of iterations: (# of iterations from 1 to 50)
    ![alt text](https://github.com/Jossome/ML-HW1/blob/master/iter=50.png?raw=true)
