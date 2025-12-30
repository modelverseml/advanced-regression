# Advanced Regression

As we all know, a regression model is commonly built using Ordinary Least Squares (OLS).

$$
RSS = \sum_{i=1}^{n} ( \hat{y_i} - y_i )^2
$$

$$
\arg \min_{\beta_0 \dots \beta_p}\ \sum_{i=1}^n\left( y_i - \beta_0 - \sum_{j=1}^p(\beta_j X_{ij}) \right)^2
$$

The objective of OLS is to estimate the coefficients β that minimize the Sum of Squared Errors (SSE).

More specifically, the OLS solution can be obtained using:
- Closed-form (analytical) solutions
- Iterative optimization methods such as Gradient Descent
<br>

**Limitations of Ordinary Least Squares**:

While OLS is simple and effective in many cases, it suffers from several important limitations:

- Multicollinearity : OLS does not account for multicollinearity. When predictors are highly correlated, the estimated coefficients become unstable and sensitive to small changes in the data.

- Overfitting : As the number of predictors increases—or when predictors are strongly correlated—the model may fit noise instead of the underlying pattern, leading to poor generalization on unseen data.


- High Variance : OLS tends to assign large coefficient values to noisy or irrelevant features. These large coefficients increase model variance, making predictions highly sensitive to minor fluctuations in the training data and reducing model stability.

This is why regularized regression techniques such as Ridge, Lasso, and Elastic Net are introduced—to control model complexity, reduce variance, and improve generalization performance.

<br>

### What is Regularization

Regularization is a technique used to prevent model overfitting by adding a penalty term to the loss function.
This penalty discourages large coefficient values, helping the model learn simpler and more generalizable patterns.

Common Regularization Techniques:
 - Lasso (Least Absolute Shrinkage and Selection Operator)
   - Also known as L1 Regularization
 - Ridge (Tikhonov Regularization)
   - Also known as L2 Regularization
 - Elastic Net
   - A combination of Lasso (L1) and Ridge (L2) regularization
<br><br>

## Lasso (Least Absolute Shrinkage and Selection Operator)

Mathematical equation

RSS + $\lambda$ * (Sum of absolute value of the magnitude of coefficients)

$$
\arg \min_{\beta_0 \dots \beta_p}\ \sum_{i=1}^n\left( y_i - \beta_0 - \sum_{j=1}^p(\beta_j X_{ij}) \right)^2 + \lambda \sum_{j=1}^p |\beta_j|
$$

- By adding the L1 regularization term, LASSO regression shrinks coefficient values toward zero.
- When the regularization parameter λ is sufficiently large, some coefficients are driven exactly to zero.
  - This happens because a larger 𝜆 increases the penalty, encouraging the optimizer to reduce or eliminate less important coefficients in order to minimize the overall objective function.
- As a result, features with zero coefficients are effectively removed from the model, making LASSO a powerful technique for feature selection.


<br><br>

## Ridge Regression (Tikhonov Regularization)

Mathematical equation:

RSS + $\lambda$ * (Sum of squares of the magnitude of coefficients)

$$
\arg \min_{\beta_0 \dots \beta_p}\ \sum_{i=1}^n\left( y_i - \beta_0 - \sum_{j=1}^p(\beta_j X_{ij}) \right)^2 + \lambda \sum_{j=1}^p \beta_j^2
$$

- By adding the L2 regularization term, Ridge regression **shrinks coefficient values toward zero** but never exactly to zero.  
- Ridge is particularly useful when dealing with **multicollinearity** (highly correlated features), as it distributes coefficient weights more evenly.  
- Unlike LASSO, Ridge does **not perform feature selection**, because no coefficients are set exactly to zero.  

<br><br>


## Elastic Net Regression

Mathematical equation:

Combination of L1 (Lasso) and L2 (Ridge) penalties:

$$
\hat{\beta} = \arg \min_{\beta_0, \dots, \beta_p} \sum_{i=1}^n \left( y_i - \beta_0 - \sum_{j=1}^p \beta_j X_{ij} \right)^2 + \lambda \left[ \alpha \sum_{j=1}^p |\beta_j| + \frac{1-\alpha}{2} \sum_{j=1}^p \beta_j^2 \right]
$$

Where:

- `λ ≥ 0` controls the **overall strength of regularization**  
- `α ∈ [0,1]` controls the **mix between L1 (Lasso) and L2 (Ridge)** penalties

**Key Points**
- Combines the benefits of **Lasso (feature selection)** and **Ridge (stability for correlated features)**.  
- Useful when there are **many correlated features**, as Lasso may arbitrarily select only one.  
- Can **shrink some coefficients to zero** while stabilizing correlated predictors.  
- Adjusting `α` balances between **sparsity** (`α → 1`) and **stability** (`α → 0`). 

