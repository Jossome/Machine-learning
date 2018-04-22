# SGD for SVM
Implement SGD for SVM for the adult income dataset. Experiment with performance as a function of the capacity parameter C.

## Files
- svm.py: the main part of algorithm for SGD for SVM, including training and testing and printing outputs.
- plot.py: run to plot the experiment result of accuracy with parameter C changing.
- plot.png: the generated image by plot.py

## Algorithm
- SGD for SVM:
```
Initialize w, b with zeros
Repeat:
    For n in 1..N:
        if y_n * (w^T dot x[n] + b) >= 1:
            w <- w - eta * 1/N * w
        else:
            w <- w - eta * (1/N * w - c * y_n * x[n])
            b <- b + eta * y_n * c
```

## Instructions
- To generate textual output, here is the sample CLI:
```
./svm.py --epochs 1 --capacity 0.868
```
- To generate plot for the experiment:
```
./plot.py
```

## Results

- Apparently, the value of capacity will not have much affect on the code performance:
```
./svm.py --epochs 1 --capacity 0.868  2.94s user 0.05s system 97% cpu 3.059 total
./svm.py --epochs 1 --capacity 80  2.99s user 0.06s system 98% cpu 3.076 total
./svm.py --epochs 1 --capacity 80000  2.93s user 0.05s system 97% cpu 3.051 total
./svm.py --epochs 1 --capacity 0.00008  3.10s user 0.05s system 99% cpu 3.186 total
```
- Experiments are conducted multiple times to find the approximately average time cost.
- It should be noticed that as the capacity reaches extremely to 0, the time cost will rise a little bit. It is probably because of the float accuracy issues.

- For epochs, we can assume that the time cost grows linearly:
```
./svm.py --epochs 1 --capacity 0.868  2.94s user 0.05s system 97% cpu 3.059 total
./svm.py --epochs 10 --capacity 0.868  5.31s user 0.07s system 98% cpu 5.453 total
./svm.py --epochs 50 --capacity 0.868  15.69s user 0.05s system 99% cpu 15.791 total
./svm.py --epochs 100 --capacity 0.868  28.58s user 0.07s system 99% cpu 28.710 total
```

## Your interpretation
Since learning rate is required to be fixed to be 0.1, we only discuss on different epochs and capacity.
- Epochs

    - Result (eta = 0.1)
    ```
    EPOCHS: 1
    CAPACITY: 0.868
    TRAINING_ACCURACY: 0.830124223602
    TEST_ACCURACY: 0.833116652878
    DEV_ACCURACY: 0.835
    
    EPOCHS: 2
    CAPACITY: 0.868
    TRAINING_ACCURACY: 0.825279503106
    TEST_ACCURACY: 0.830752866091
    DEV_ACCURACY: 0.8365
    
    EPOCHS: 3
    CAPACITY: 0.868
    TRAINING_ACCURACY: 0.831552795031
    TEST_ACCURACY: 0.83276208486
    DEV_ACCURACY: 0.836875
    
    EPOCHS: 4
    CAPACITY: 0.868
    TRAINING_ACCURACY: 0.82248447205
    TEST_ACCURACY: 0.828743647323
    DEV_ACCURACY: 0.833
    
    EPOCHS: 5
    CAPACITY: 0.868
    TRAINING_ACCURACY: 0.831677018634
    TEST_ACCURACY: 0.834889492968
    DEV_ACCURACY: 0.835375
    ```
    
    - We can see that as epochs increases, the accuracy is fluctuating instead of improving.
    - This is because the learning rate is too large, even if it is 0.1.
    - OK, let's make learning rate smaller (just in this case). Let's say learning rate is 0.01.
   
    - Result (eta = 0.01)
    ```
    EPOCHS: 1
    CAPACITY: 0.868
    TRAINING_ACCURACY: 0.844720496894
    TEST_ACCURACY: 0.847890320293
    DEV_ACCURACY: 0.848875
    
    EPOCHS: 2
    CAPACITY: 0.868
    TRAINING_ACCURACY: 0.84602484472
    TEST_ACCURACY: 0.846235669543
    DEV_ACCURACY: 0.849625
    
    EPOCHS: 3
    CAPACITY: 0.868
    TRAINING_ACCURACY: 0.846211180124
    TEST_ACCURACY: 0.845171965489
    DEV_ACCURACY: 0.849375
    
    EPOCHS: 4
    CAPACITY: 0.868
    TRAINING_ACCURACY: 0.84801242236
    TEST_ACCURACY: 0.846235669543
    DEV_ACCURACY: 0.85025
    
    EPOCHS: 5
    CAPACITY: 0.868
    TRAINING_ACCURACY: 0.846770186335
    TEST_ACCURACY: 0.846353858882
    DEV_ACCURACY: 0.8495
    ```

    - We can see that the accuracy reaches peak when epochs = 4. Or we can say the model converges when epochs reaches 4.
    - Also we can naively assume that a smaller learning rate will generate higher accuracy, at the cost of more epochs.

- Capacity
    - Note that in the objective function, the part that capacity is multiplied to decides how well the model fits.
    - If we give too much weight on this part, the model may go overfitted.
    - Let's take the result in the plot.png and discuss.
        - The curve looks like a parabola when c <= 10. This parabola part is the meaningful part, because the value of c is not very big so it's less possible for the model to overfit.
        - The curve reaches its peak when c = 0.0695.
        - The curve starts to fluctuate when c > 10. This part is unpredictable, because a large c gives the model too much weight on fitting the training data. It is hard to say if the performance would still be good on test or dev datasets.
    
- Conclusion
    - It's not true that the more epochs the better. The model will converge after certain number of epochs, depending on learning rate. The smaller learning rate is, the more epochs it takes to converge.
    - Capacity is critical. In this case, a capacity less than 10 is acceptable, and c = 0.0695 may lead to the best model. If the capacity is too large, the performance of the model is highly unpredictable, and overfitting issue may occur.

## References
https://docs.scipy.org/doc/
https://matplotlib.org/api/pyplot_api.html
