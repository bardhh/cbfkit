# How to expand derivative B?

Since
$$
\begin{aligned}
B(xr,xo) = \frac{(xr_0-xo_0)^2}{a^2} + \frac{(xr_1-xo_1)^2}{b^2} - 1 \tag{1}
\end{aligned}
$$
$$
\begin{aligned}
\dot{B}(xr,xo) = \frac{\partial B}{\partial xr} \cdot \dot{xr} + \frac{\partial B}{\partial xo} \cdot \dot{xo} \tag{2}
\end{aligned}
$$

then
$$
\begin{aligned}
\frac{\partial{B}}{\partial{xr}} = \frac{\partial{B(xr,xo)}}{\partial{xr}} &= \frac{\partial}{\partial{xr}} (\frac{(xr_0-xo_0)^2}{a^2} + \frac{(xr_1-xo_1)^2}{b^2} - 1)\\
&= \frac{\partial}{\partial{xr}} (\frac{(xr_0^2-2xr_0xo_0+xo_0^2)}{a^2} + \frac{(xr_1^2-2xr_1xo_1+xo_1^2)}{b^2} - 1)\\
&= \begin{bmatrix}
        \frac{\partial}{\partial{xr_0}} (\frac{(xr_0^2-2xr_0xo_0+xo_0^2)}{a^2} + \frac{(xr_1^2-2xr_1xo_1+xo_1^2)}{b^2} - 1)\\
        \frac{\partial}{\partial{xr_1}} (\frac{(xr_0^2-2xr_0xo_0+xo_0^2)}{a^2} + \frac{(xr_1^2-2xr_1xo_1+xo_1^2)}{b^2} - 1)
   \end{bmatrix}^T\\
&= \begin{bmatrix}
        \frac{(2xr_0-2xo_0)}{a^2})\\
        \frac{(2xr_1-2xo_1)}{b^2})
   \end{bmatrix}^T
= \begin{bmatrix}
        \frac{2}{a^2}(xr_0 - xo_0)\\
        \frac{2}{b^2}(xr_1 - xo_1)
   \end{bmatrix}^T
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial{B}}{\partial{xo}} = \frac{\partial{B(xr,xo)}}{\partial{xo}} &= \frac{\partial}{\partial{xo}} (\frac{(xr_0-xo_0)^2}{a^2} + \frac{(xr_1-xo_1)^2}{b^2} - 1)\\
&= \frac{\partial}{\partial{xo}} (\frac{(xr_0^2-2xr_0xo_0+xo_0^2)}{a^2} + \frac{(xr_1^2-2xr_1xo_1+xo_1^2)}{b^2} - 1)\\
&= \begin{bmatrix}
        \frac{\partial}{\partial{xo_0}} (\frac{(xr_0^2-2xr_0xo_0+xo_0^2)}{a^2} + \frac{(xr_1^2-2xr_1xo_1+xo_1^2)}{b^2} - 1)\\
        \frac{\partial}{\partial{xo_1}} (\frac{(xr_0^2-2xr_0xo_0+xo_0^2)}{a^2} + \frac{(xr_1^2-2xr_1xo_1+xo_1^2)}{b^2} - 1)
   \end{bmatrix}^T\\
&= \begin{bmatrix}
        \frac{(-2ro_0+2xo_0)}{a^2})\\
        \frac{(-2ro_1+2xo_1)}{b^2})
   \end{bmatrix}^T
= \begin{bmatrix}
        \frac{2}{a^2}(xo_0 - xr_0)\\
        \frac{2}{b^2}(xo_1 - xr_1)
   \end{bmatrix}^T
\end{aligned}
$$

Finally
$$
\begin{aligned}
\dot{B}(xr,xo) =
        \begin{bmatrix}
                \frac{2}{a^2} (xr_0 - xo_0) \\
                \frac{2}{b^2} (xr_1 - xo_1)
        \end{bmatrix}^T
        \cdot \dot{xr} +
        \begin{bmatrix}
                \frac{2}{a^2} (xo_0 - xr_0) \\
                \frac{2}{b^2} (xo_1 - xr_1)
        \end{bmatrix}^T
        \cdot \dot{xo} \tag{3}
\end{aligned}
$$