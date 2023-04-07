# How to pose CBF control problem to quadric programing?

Now we have control problem with CBF
$$
\begin{aligned}
\min_{u} \quad & ||u_{ref} - u||\\
\textrm{s.t.} \quad & \begin{bmatrix} \frac{2}{a^2}(x_0 - z_0) \\ \frac{2}{b^2}(x_1 - z_1)\end{bmatrix} \dot{x} \ge -(\frac{(x_0-z_0)^2}{a^2} + \frac{(x_1-z_1)^2}{b^2} - 1) \\
... \tag{5}
\end{aligned}
$$

The cost function $||u_{ref} - u||$ is the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance), and here we have two dimensions. Then,
$$
\begin{aligned}
||u_{ref} - u|| &= (u_{ref0}-u_0)^2 + (u_{ref1}-u_1)^2 \\
&= (u_{ref0}^2-2u_{ref0}u_0+u_0^2) + (u_{ref1}^2-2u_{ref1}u_1+u_1^2) \\
&= \begin{bmatrix}u_0 & u_1\end{bmatrix} \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix} \begin{bmatrix}u_0 \\ u_1\end{bmatrix} -2 \begin{bmatrix}u_{ref0} & u_{ref1}\end{bmatrix} \begin{bmatrix}u_0 \\ u_1\end{bmatrix} + \begin{bmatrix}u_{ref0} & u_{ref1}\end{bmatrix} \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix} \begin{bmatrix}u_{ref0} \\ u_{ref1}\end{bmatrix}\\
&\approx \begin{bmatrix}u_0 & u_1\end{bmatrix} \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix} \begin{bmatrix}u_0 \\ u_1\end{bmatrix} -2 \begin{bmatrix}u_{ref0} & u_{ref1}\end{bmatrix} \begin{bmatrix}u_0 \\ u_1\end{bmatrix} \\
&= \dot{x}^T \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix} \dot{x} -2 \begin{bmatrix}u_{ref0} \\ u_{ref1}\end{bmatrix}^T \dot{x} \\
&= 2(\frac{1}{2} \dot{x}^T I_2 \dot{x} + \begin{bmatrix}-u_{ref0} \\ -u_{ref1}\end{bmatrix}^T \dot{x}) \approx \frac{1}{2} \dot{x}^T I_2 \dot{x} + \begin{bmatrix}-u_{ref0} \\ -u_{ref1}\end{bmatrix}^T \dot{x}
\end{aligned}
$$

Note that $u_{ref0}$ and $u_{ref1}$ are constants and therefore ommited in the case when they are not multiplying the decision variables. Doing so does not affect the solution of the optmiziation problem.

Next, the direction of the inequality sign needs to be changed.
$$
\begin{aligned}
& \begin{bmatrix} \frac{2}{a^2}(x_0 - z_0) \\ \frac{2}{b^2}(x_1 - z_1)\end{bmatrix} \dot{x} \ge -(\frac{(x_0-z_0)^2}{a^2} + \frac{(x_1-z_1)^2}{b^2} - 1) \\
\equiv &\begin{bmatrix} \frac{-2}{a^2}(x_0 - z_0) \\ \frac{-2}{b^2}(x_1 - z_1)\end{bmatrix} \dot{x} \le (\frac{(x_0-z_0)^2}{a^2} + \frac{(x_1-z_1)^2}{b^2} - 1)
\end{aligned}
$$

Finally we can transform (12) as,
$$
\begin{aligned}
\min_{u} \quad & \frac{1}{2} \dot{x}^T I_2 \dot{x} + \begin{bmatrix}-u_{ref0} \\ -u_{ref1}\end{bmatrix}^T \dot{x}\\
\textrm{s.t.} \quad & \begin{bmatrix} \frac{-2}{a^2}(x_0 - z_0) \\ \frac{-2}{b^2}(x_1 - z_1)\end{bmatrix} \dot{x} \le (\frac{(x_0-z_0)^2}{a^2} + \frac{(x_1-z_1)^2}{b^2} - 1) \\
...
\end{aligned}
$$

While compairing above with a quadratic problem of the form,
$$
\begin{aligned}
\min \quad & \frac{1}{2}\mathbf{x}^T P\mathbf{x} + q^T\mathbf{x} \\　\textrm{s.t.} \quad & G\mathbf{x} \le h \\　\quad & A\mathbf{x} = b \tag{6}
\end{aligned}
$$
We can see,
$$
\begin{aligned}
    \mathbf{x} = \dot{x} = \begin{bmatrix}u_0 \\ u_1\end{bmatrix}
\end{aligned}
$$
is the decision variable, where,
$$
\begin{aligned}
P = I_2 =   \begin{bmatrix}
                1 & 0\\
                0 & 1
            \end{bmatrix} \tag{7}
\end{aligned}
$$
$$
\begin{aligned}
q = \begin{bmatrix}
        -u_{ref0}\\
        -u_{ref1}
    \end{bmatrix} \tag{8}
\end{aligned}
$$
$$
\begin{aligned}
G = \begin{bmatrix}
        \frac{-2}{a^2}(x_0 - z_0)\\
        \frac{-2}{b^2}(x_1 - z_1)
    \end{bmatrix}^T \tag{9}
\end{aligned}
$$
$$
\begin{aligned}
h = (\frac{(x_0-z_0)^2}{a^2} + \frac{(x_1-z_1)^2}{b^2} - 1) \tag{10}
\end{aligned}
$$
