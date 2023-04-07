$$
\begin{aligned}
L_fB(x) &= \frac{\partial B}{\partial x}\cdot f(x)\\
&=  \begin{bmatrix}
        \frac{2}{a^2}(xr_0 - z_0)\\
        \frac{2}{b^2}(xr_1 - z_1)\\
        0
    \end{bmatrix}^T
\cdot
\begin{bmatrix}
    cos(xr_2)\\
    sin(xr_2)\\
    0
\end{bmatrix}\\
&=   \frac{2}{a^2}(xr_0 - z_0)\cdot cos(xr_2)
    +\frac{2}{b^2}(xr_1 - z_1)\cdot sin(xr_2)
\end{aligned}
$$

$$
\begin{aligned}
L^2_fB(x) &= \frac{\partial}{\partial x}(L_fB(x))^T \cdot f(x) \\
&= \frac{\partial}{\partial x}(\frac{2}{a^2}(xr_0 - z_0)\cdot cos(xr_2)+\frac{2}{b^2}(xr_1 - z_1)\cdot sin(xr_2)) \cdot f(x)\\
&= \begin{bmatrix}
        \frac{\partial}{\partial xr_0}(\frac{2}{a^2}(xr_0 - z_0)\cdot cos(xr_2)+\frac{2}{b^2}(xr_1 - z_1)\cdot sin(xr_2))\\
        \frac{\partial}{\partial xr_1}(\frac{2}{a^2}(xr_0 - z_0)\cdot cos(xr_2)+\frac{2}{b^2}(xr_1 - z_1)\cdot sin(xr_2))\\
        \frac{\partial}{\partial xr_2}(\frac{2}{a^2}(xr_0 - z_0)\cdot cos(xr_2)+\frac{2}{b^2}(xr_1 - z_1)\cdot sin(xr_2))\\
    \end{bmatrix}^T
    \cdot f(x) \\
&= \begin{bmatrix}
        \frac{2}{a^2}cos(xr_2)\\
        \frac{2}{b^2}sin(xr_2)\\
        -\frac{2}{a^2}(xr_0 - z_0)\cdot sin(xr_2)+\frac{2}{b^2}(xr_1 - z_1)\cdot cos(xr_2)\\
    \end{bmatrix}^T
    \cdot
    \begin{bmatrix}
        cos(xr_2)\\
        sin(xr_2)\\
        0
    \end{bmatrix}\\
&= \frac{2}{a^2}cos^2(xr_2) + \frac{2}{b^2}sin^2(xr_2)
\tag{4}
\end{aligned}
$$

$$
\begin{aligned}
L_gL_fB(x) &= \frac{\partial}{\partial x}(L_fB(x))^T \cdot g(x) \\
&= \begin{bmatrix}
        \frac{2}{a^2}cos(xr_2)\\
        \frac{2}{b^2}sin(xr_2)\\
        -\frac{2}{a^2}(xr_0 - z_0)\cdot sin(xr_2)+\frac{2}{b^2}(xr_1 - z_1)\cdot cos(xr_2)\\
    \end{bmatrix}^T
    \cdot
    \begin{bmatrix}
        0\\
        0\\
        1
    \end{bmatrix}\\
&= -\frac{2}{a^2}(xr_0 - z_0)\cdot sin(xr_2)+\frac{2}{b^2}(xr_1 - z_1)\cdot cos(xr_2) \tag{5}
\end{aligned}
$$