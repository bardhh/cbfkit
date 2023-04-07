# How to expand derivative B?

Since
$$
\begin{aligned}
B(x) = \frac{(x_0-z_0)^2}{a^2} + \frac{(x_1-z_1)^2}{b^2} - 1 \tag{1}
\end{aligned}
$$

then
$$
\begin{aligned}
\frac{\partial{B}}{\partial{x}} = \frac{\partial{B(x)}}{\partial{x}} &= \frac{\partial}{\partial{x}} (\frac{(x_0-z_0)^2}{a^2} + \frac{(x_1-z_1)^2}{b^2} - 1)\\
&= \frac{\partial}{\partial{x}} (\frac{(x_0^2-2z_0x_0+z_0^2)}{a^2} + \frac{(x_1^2-2z_1x_1+z_1^2)}{b^2} - 1)\\
&= \begin{bmatrix}
        \frac{\partial}{\partial{x_0}} (\frac{(x_0^2-2z_0x_0+z_0^2)}{a^2} + \frac{(x_1^2-2z_1x_1+z_1^2)}{b^2} - 1)\\
        \frac{\partial}{\partial{x_1}} (\frac{(x_0^2-2z_0x_0+z_0^2)}{a^2} + \frac{(x_1^2-2z_1x_1+z_1^2)}{b^2} - 1)
   \end{bmatrix}^T\\
&= \begin{bmatrix}
        \frac{(2x_0-2z_0)}{a^2})\\
        \frac{(2x_1-2z_1)}{b^2})
   \end{bmatrix}^T
= \begin{bmatrix}
        \frac{2}{a^2}(x_0 - z_0)\\
        \frac{2}{b^2}(x_1 - z_1)
   \end{bmatrix}^T 　\tag{4}
\end{aligned}
$$
