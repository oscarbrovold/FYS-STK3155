## Week 35

### Exercise 2

##### Discussion of the meaning of $R^2$-score and MSE

##### The $R^2$-score
The $R^2$-score is a a measurement of the quality of our model in relation to the ability to correctly
predict future data. The score typically ranges from 0.0 to 1.0, 1.0 meaning we have a perfect fit, the worst is arbitrary low, most likely due to a wrong choise of model. If our $R^2$ score is x, we can interpent this as our model accounting for $x\times 100$% of the variance in the dependent variable. 

Our score of $0.9965$ is therefore a high score, and our model is likely to predict future data. In spesific our model accounts for $99.65$% of the variance we observe in the dependent variable.  

##### The mean squared error

The MSE, or mean squared error, is also a metric used to determine the quality of the model. In spesific, by using MSE, we calculate the average of the squared errors. The error is the diffrence between our predicted point and the acutal data. A score of 0 means there is no error between our model and our data, meaning our model explains the data perfectly. A higher score indicates a poorer fit. 

Our MSE is $0.0078$, this is a good score, and our model does a good job explaining our data.

##### Variation in the coefficient in front of the added stochastic noise term

When applying more noise to the data we get as expected, a higher MSE and a lower $R^2$-score. A lower $R^2$-score indicates that the models ability to explain the variance in the dependent variable gets worse. A higher MSE explains that average error has gotten worse. All in all indicating a worse fit, and a poorer model.  

### Exercise 3

##### Discussion of the MSE (of training and test) as functions of the complexity

![image](https://github.com/user-attachments/assets/e856b32c-46e9-4260-90d5-37e7ddb65544)

*Figure 1: Test and training data MSE as functions of the polynomial degree.*

There is a good correlation between the MSE for both training and test data at the complexity 8, in addition the MSE is low. I would therefore argue that a polynomial degree of 8 would be optimal without the risk of overfitting.

