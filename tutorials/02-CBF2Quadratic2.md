# How to pose CBF control problem to quadric programing?

Now we have control problem with CBF
$$
\begin{aligned}
\min_{u} \quad & ||u_{ref} - u||\\
\textrm{s.t.} \quad & \begin{bmatrix} \frac{2}{a^2} (xr_0 - xo_0) \\  \frac{2}{b^2} (xr_1 - xo_1) \\ \frac{2}{a^2} (xo_0 - xr_0) \\  \frac{2}{b^2} (xo_1 - xr_1) \end{bmatrix} \cdot \begin{bmatrix}u_0 \\ u_1 \\ -1 \\ 0\end{bmatrix} \ge -(\frac{(xr_0-xo_0)^2}{a^2} + \frac{(xr_1-xo_1)^2}{b^2} - 1) \tag{4}
\end{aligned}
$$

The cost function $||u_{ref} - u||$ is the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance), and here we have four dimensions. Then,
$$
\begin{aligned}
||u_{ref} - u|| &= (u_{ref0}-u_0)^2 + (u_{ref1}-u_1)^2 \\
&= (u_{ref0}^2-2u_{ref0}u_0+u_0^2) + (u_{ref1}^2-2u_{ref1}u_1+u_1^2) + (u_{ref2}^2-2u_{ref2}u_2+u_2^2) + (u_{ref3}^2-2u_{ref3}u_3+u_3^2)\\
&= \begin{bmatrix}u_0 & u_1 & u_2 & u_3 \end{bmatrix} \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix}u_0 \\ u_1 \\ u_2 \\ u_3 \end{bmatrix} -2 \begin{bmatrix}u_{ref0} & u_{ref1} & u_{ref2} & u_{ref3}\end{bmatrix} \begin{bmatrix}u_0 \\ u_1 \\ u_2 \\ u_3 \end{bmatrix} + \begin{bmatrix}u_{ref0} & u_{ref1} & u_{ref2} & u_{ref3}\end{bmatrix} \begin{bmatrix}1 & 0 & 0 & 0\\ 0 & 1 & 0 & 0\\ 0 & 0 & 1 & 0\\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix}u_{ref0} \\ u_{ref1} \\ u_{ref2} \\ u_{ref3}\end{bmatrix}\\
&\approx \begin{bmatrix}u_0 & u_1 & u_2 & u_3 \end{bmatrix} \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix}u_0 \\ u_1 \\ u_2 \\ u_3 \end{bmatrix} -2 \begin{bmatrix}u_{ref0} & u_{ref1} & u_{ref2} & u_{ref3}\end{bmatrix} \begin{bmatrix}u_0 \\ u_1 \\ u_2 \\ u_3 \end{bmatrix} \\
&= \dot{x}^T \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \dot{x} -2 \begin{bmatrix}u_{ref0} \\ u_{ref1} \\ u_{ref2} \\ u_{ref3}\end{bmatrix}^T \dot{x} \\
&= 2(\frac{1}{2} \dot{x}^T I_4 \dot{x} + \begin{bmatrix}-u_{ref0} \\ -u_{ref1} \\ -u_{ref2} \\ -u_{ref3}\end{bmatrix}^T \dot{x}) \approx \frac{1}{2} \dot{x}^T I_4 \dot{x} + \begin{bmatrix}-u_{ref0} \\ -u_{ref1} \\ -u_{ref2} \\ -u_{ref3}\end{bmatrix}^T \dot{x}
\end{aligned}
$$

Note that $u_{ref0}$, $u_{ref1}$, $u_{ref2}$, $u_{ref3}$ are constants and therefore ommited in the case when they are not multiplying the decision variables. Doing so does not affect the solution of the optmiziation problem.

Next, the direction of the inequality sign needs to be changed.

$$
\begin{aligned}
& \begin{bmatrix} \frac{2}{a^2} (xr_0 - xo_0) \\  \frac{2}{b^2} (xr_1 - xo_1) \\ \frac{2}{a^2} (xo_0 - xr_0) \\  \frac{2}{b^2} (xo_1 - xr_1) \end{bmatrix} \cdot \begin{bmatrix}u_0 \\ u_1 \\ -1 \\ 0\end{bmatrix} &\ge& -(\frac{(xr_0-xo_0)^2}{a^2} + \frac{(xr_1-xo_1)^2}{b^2} - 1)\\
\equiv & \begin{bmatrix} \frac{2}{a^2} (xr_0 - xo_0) \\  \frac{2}{b^2} (xr_1 - xo_1) \\ \frac{2}{a^2} (xo_0 - xr_0) \\  \frac{2}{b^2} (xo_1 - xr_1) \end{bmatrix} \cdot \dot{x} &\ge& -(\frac{(xr_0-xo_0)^2}{a^2} + \frac{(xr_1-xo_1)^2}{b^2} - 1)\\
\equiv &\begin{bmatrix} \frac{2}{a^2} (xr_0 - xo_0) \\  \frac{2}{b^2} (xr_1 - xo_1) \\ \frac{2}{a^2} (xo_0 - xr_0) \\  \frac{2}{b^2} (xo_1 - xr_1) \end{bmatrix} \cdot \dot{x} &\le& (\frac{(xr_0-xo_0)^2}{a^2} + \frac{(xr_1-xo_1)^2}{b^2} - 1)
\end{aligned}
$$

Finally we can transform (11) as,
$$
\begin{aligned}
\min_{u} \quad & \frac{1}{2} \dot{x}^T I_4 \dot{x} + \begin{bmatrix}-u_{ref0} \\ -u_{ref1} \\ -u_{ref2} \\ -u_{ref3}\end{bmatrix}^T \dot{x}\\
\textrm{s.t.} \quad & \begin{bmatrix} \frac{2}{a^2} (xr_0 - xo_0) \\  \frac{2}{b^2} (xr_1 - xo_1) \\ \frac{2}{a^2} (xo_0 - xr_0) \\  \frac{2}{b^2} (xo_1 - xr_1) \end{bmatrix} \cdot \dot{x} \le (\frac{(xr_0-xo_0)^2}{a^2} + \frac{(xr_1-xo_1)^2}{b^2} - 1)
\end{aligned}
$$

While compairing above with a quadratic problem of the form,
$$
\begin{aligned}
\min \quad & \frac{1}{2}\mathbf{x}^T P\mathbf{x} + q^T\mathbf{x} \\
\textrm{s.t.} \quad & G\mathbf{x} \le h \tag{5}
\end{aligned}
$$
We can see,
$$
\begin{aligned}
    \mathbf{x} = \dot{x} = \begin{bmatrix}u_0 \\ u_1 \\ u_2 \\ u_3\end{bmatrix} = \begin{bmatrix}u_0 \\ u_1 \\ -1 \\ 0\end{bmatrix}
\end{aligned}
$$
where $u_2$, $u_3$ are uncontrolable and predefined as the obstacle goes to straight forward.
$$
\begin{aligned}
P = I_4 = \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \tag{6}
\end{aligned}
$$
$$
\begin{aligned}
q = \begin{bmatrix} -u_{ref0} \\ -u_{ref1} \\ -u_{ref2} \\ -u_{ref3} \end{bmatrix} = \begin{bmatrix} -u_{ref0} \\ -u_{ref1} \\ 0 \\ 0 \end{bmatrix} \tag{7}
\end{aligned}
$$
where $u_{ref2}$, $u_{ref3}$ are uncontrolable because of an obstacle's reference parameters.
$$
\begin{aligned}
G = \begin{bmatrix} \frac{2}{a^2} (xr_0 - xo_0) \\  \frac{2}{b^2} (xr_1 - xo_1) \\ \frac{2}{a^2} (xo_0 - xr_0) \\  \frac{2}{b^2} (xo_1 - xr_1) \end{bmatrix} \tag{8}
\end{aligned}
$$
$$
\begin{aligned}
h = (\frac{(xr_0-xo_0)^2}{a^2} + \frac{(xr_1-xo_1)^2}{b^2} - 1) \tag{9}
\end{aligned}
$$
