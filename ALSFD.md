## 定理：各向异性水平集标架扩散与曲面 Bochner 热流的等价性

**设定**
令 \( \Omega \subset \mathbb{R}^3 \) 为开集，\( f \in C^\infty(\Omega) \) 是定义在 \(\Omega\) 上的**符号距离函数**，满足
\[
|\nabla f| = 1,
\]
且其水平集 \( \mathcal{M}_c = \{x \in \Omega : f(x) = c\} \) 均为光滑、紧致、可定向的二维曲面（无边界或边界在 \(\Omega\) 外）。
记单位法向量场
\[
\mathbf{n}(x) = \nabla f(x),
\]
切空间的正交投影矩阵为
\[
\mathbf{P}(x) = \mathbf{I} - \mathbf{n}(x)\mathbf{n}(x)^\top .
\]

对于任意向量场 \(\mathbf{v} \in C^1(\Omega, \mathbb{R}^3)\)，定义算子
\[
\mathcal{L}(\mathbf{v}) = \mathbf{P} \big( \nabla \cdot (\mathbf{P} \nabla \mathbf{v}) \big),
\]
其中梯度与散度均为 \(\mathbb{R}^3\) 中的标准欧氏运算。

---

### 主要结果

**定理 1（空间退化扩散与曲面内蕴扩散的等价性）**
若 \(\mathbf{v}\) 是处处切于水平集的向量场，即 \(\mathbf{P}\mathbf{v} = \mathbf{v}\)，则在每一个水平集 \(\mathcal{M}_c\) 上有
\[
\mathcal{L}(\mathbf{v})\big|_{\mathcal{M}_c} = \Delta_{B,\mathcal{M}_c} \big( \mathbf{v}|_{\mathcal{M}_c} \big),
\]
其中 \(\Delta_{B,\mathcal{M}_c}\) 是曲面 \(\mathcal{M}_c\) 上的 **Bochner 拉普拉斯**（亦称投影拉普拉斯或粗糙拉普拉斯），其定义为：对于切向量场 \(\mathbf{u} \in \mathfrak{X}(\mathcal{M}_c)\)，
\[
\Delta_{B,\mathcal{M}_c} \mathbf{u} = \mathbf{P} \big( \Delta_{\mathcal{M}_c} \mathbf{u} \big),
\]
而 \(\Delta_{\mathcal{M}_c} \mathbf{u}\) 指对 \(\mathbf{u}\) 的每一个笛卡尔分量（作为 \(\mathcal{M}_c\) 上的光滑函数）施加 Laplace–Beltrami 算子。

**推论 1（演化方程的切向保持性与解耦）**
设初始向量场 \(\mathbf{v}_0\) 满足 \(\mathbf{P} \mathbf{v}_0 = \mathbf{v}_0\)，\(\mathbf{v}(t,x)\) 是初值问题
\[
\frac{\partial \mathbf{v}}{\partial t} = \mathcal{L}(\mathbf{v}), \qquad \mathbf{v}(0) = \mathbf{v}_0
\]
的解。则对于所有 \(t \ge 0\)，\(\mathbf{v}\) 始终保持 \(\mathbf{P}\mathbf{v} = \mathbf{v}\)，且其在每一个水平集 \(\mathcal{M}_c\) 上的限制独立地按该曲面的 Bochner 热方程演化：
\[
\frac{\partial}{\partial t} \big( \mathbf{v}|_{\mathcal{M}_c} \big) = \Delta_{B,\mathcal{M}_c} \big( \mathbf{v}|_{\mathcal{M}_c} \big).
\]

---

### 证明

证明的关键是如下在水平集邻域内处处成立的代数恒等式。

**引理 1**
对于任意 \(C^2\) 向量场 \(\mathbf{v}: \Omega \to \mathbb{R}^3\)，在 \(\Omega\) 上恒有
\[
\nabla \cdot (\mathbf{P} \nabla \mathbf{v}) = \Delta_{\mathcal{M}} \mathbf{v},
\]
其中 \(\Delta_{\mathcal{M}} \mathbf{v}\) 表示在每个水平集 \(\mathcal{M}_{f(x)}\) 上对 \(\mathbf{v}\) 的每个分量施加 Laplace–Beltrami 算子。

*引理的证明*
由 \(\mathbf{P} = \mathbf{I} - \mathbf{n}\mathbf{n}^\top\)，采用分量记法（爱因斯坦求和约定）。对第 \(j\) 个分量，
\[
\big( \nabla \cdot (\mathbf{P} \nabla \mathbf{v}) \big)_j = \partial_i \big( P_{ik} \, \partial_k v_j \big)
= (\partial_i P_{ik}) \partial_k v_j + P_{ik}\, \partial_i \partial_k v_j.
\]
因为 \(|\mathbf{n}| = 1\)，有 \(\mathbf{n}\cdot \partial_i \mathbf{n} = 0\)，且 \(\nabla \cdot \mathbf{n} = \Delta f = -2H\)，其中 \(H\) 是各水平集的平均曲率。于是
\[
\partial_i P_{ik} = \partial_i(\delta_{ik} - n_i n_k) = -\partial_i n_i \, n_k - n_i \partial_i n_k = -(\Delta f) n_k = 2H n_k.
\]
第一项成为 \(2H n_k \partial_k v_j = 2H (\partial_{\mathbf{n}} \mathbf{v})_j\)。
第二项：\(P_{ik} \partial_i \partial_k v_j = \Delta v_j - n_i n_k \partial_i \partial_k v_j = \Delta v_j - \partial_{\mathbf{n}}^2 v_j\)。
因此
\[
\nabla \cdot (\mathbf{P} \nabla \mathbf{v}) = \Delta \mathbf{v} - \partial_{\mathbf{n}}^2 \mathbf{v} + 2H \partial_{\mathbf{n}} \mathbf{v}. \tag{1}
\]

现在，在法向坐标 \((y, c)\) 下，其中 \(c = f(x)\)，三维欧氏拉普拉斯分裂为（作用于标量函数）：
\[
\Delta = \Delta_{\mathcal{M}_c} + \partial_c^2 - 2H \partial_c .
\]
此公式对向量场的每个分量依然成立，故
\[
\Delta \mathbf{v} = \Delta_{\mathcal{M}} \mathbf{v} + \partial_{\mathbf{n}}^2 \mathbf{v} - 2H \partial_{\mathbf{n}} \mathbf{v}. \tag{2}
\]

将 (2) 代入 (1)，\(\partial_{\mathbf{n}}^2 \mathbf{v}\) 与 \(-2H \partial_{\mathbf{n}} \mathbf{v}\) 精确抵消，即得
\[
\nabla \cdot (\mathbf{P} \nabla \mathbf{v}) = \Delta_{\mathcal{M}} \mathbf{v}.
\]
引理得证。 ∎

*定理 1 的证明*
对切向场 \(\mathbf{v}\)（\(\mathbf{P}\mathbf{v} = \mathbf{v}\)），由引理，
\[
\mathcal{L}(\mathbf{v}) = \mathbf{P} \big( \nabla \cdot (\mathbf{P} \nabla \mathbf{v}) \big) = \mathbf{P} \big( \Delta_{\mathcal{M}} \mathbf{v} \big).
\]
限制在水平集 \(\mathcal{M}_c\) 上，右侧正是曲面 Bochner 拉普拉斯的定义 \(\Delta_{B,\mathcal{M}_c}(\mathbf{v}|_{\mathcal{M}_c})\)。 ∎

*推论 1 的证明*
由定理 1，演化方程右端在每点均位于切空间内（因为左乘 \(\mathbf{P}\)），故若初始场切向，则对所有时间导数均为切向，积分得切向性保持。进一步，该方程在每一水平集上是封闭的 Bochner 热方程，不同水平集之间无耦合，因为 \(\mathcal{L}(\mathbf{v})\) 不包含任何法向导数。 ∎

---

### 评注

1. **曲率自动入账**
该证明表明，退化扩散算子 \(\nabla \cdot (\mathbf{P} \nabla \mathbf{v})\) 本身就完美地包含了所有因曲面弯曲造成的曲率效应，无需额外补偿项。我之前的批评——即退化扩散缺少曲率耦合——是错误的；严格的微分恒等式彻底解决了这一疑虑。

2. **无需“投影”插值**
方程 \(\partial \mathbf{v}/\partial t = \mathbf{P} \nabla \cdot (\mathbf{P} \nabla \mathbf{v})\) 完全在空间中定义，不依赖任何显式的最近点查询或向曲面的插值。这实现了“用体积 PDE 内生表面内蕴几何”的目标，且全程可微。

3. **与原始 ALSFD 的关系**
该定理为修正后的 ALSFD 提供了坚实的理论地基：只需选择符号距离函数 \(f\)，构建投影 \(\mathbf{P}\)，求解上述 PDE，即可在**所有平行水平集**上同步获得彼此独立但均满足 Bochner 热流的切向量场。这正是原始构想所追求的“升维等价性”。

4. **实际离散与窄带化**
在数值实现中，我们只需在围绕目标曲面的窄带内求解该 PDE，窄带边界可采用吸收层或零法向梯度条件。由于等值面之间解耦，窄带解法不会引入伪耦合。

于是，修正后的 ALSFD 不再是一种有缺陷的构想，而是一个被严格证明的、连接空间退化椭圆算子与曲面 Bochner 算子的几何分析事实。
