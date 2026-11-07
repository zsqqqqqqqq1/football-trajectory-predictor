# 三轴自旋模型与验证

在惯性系中积分球心位置 \(\mathbf r\)、速度 \(\mathbf v\) 和角速度 \(\boldsymbol\omega\)。设 \(\mathbf u=\mathbf v-\mathbf w\)，其中 \(\mathbf w\) 为风速，命令行默认静止空气。模型为

\[
\dot{\mathbf r}=\mathbf v,\qquad
\dot{\mathbf v}=-g\mathbf e_z-\frac{\rho\pi R^2}{2m}C_D(|\mathbf u|)|\mathbf u|\mathbf u
                  + C_M(\boldsymbol\omega\times\mathbf u),\qquad
\dot{\boldsymbol\omega}=-\mu\boldsymbol\omega .
\]

\(C_M=8\pi\rho R^3/(3m)\) 是加速度式中乘在叉积前的无量纲常数。指数衰减改变自旋大小，轴向在惯性系中保持固定。马格努斯加速度的方向随相对速度改变。DOP853 定位第一次向前穿越目标平面或下降到 \(z=R\) 的事件。

## 固定气动输入

- \(C_D\) 由相对空速的 PCHIP 插值得到，表外取端点。默认曲线是 Asai 等（2026）图 3 的 Trionda A 稀疏近似，节点直接记录在 `model.py` 和预测输出中。球面朝向和旋转对实际阻力的影响没有纳入此固定曲线的预测区间。[风洞数据来源](https://doi.org/10.1038/s41598-026-56275-1)
- `--cd-profile team471` 使用 Team 471 附录的光滑球阻力曲线，作为敏感性对照。该队第 9–10 页给出 \(C_D\) 随速度的插值方法。[2022 Team 471](https://www.uphysicsc.com/2022-GM-B-471.pdf)
- 三维马格努斯加速度及线性阻力矩来自 Team 480 第 5–6 页，Team 358 §3.1.3 也给出指数自旋衰减。默认 \(\mu=0.10\,\mathrm{s}^{-1}\) 是可调整的初始假设，可用 `--spin-decay 0` 做无衰减对照。[2022 Team 480](https://www.uphysicsc.com/2022-GM-B-480.pdf) · [2022 Team 358](https://www.uphysicsc.com/2022-GM-B-358.pdf)

## 反演与滤波

MCMC 拟合 \((v_{x0},v_{y0},v_{z0},\omega_{x0},\omega_{y0},\omega_{z0})\)。用 \((\mathbf r_2-\mathbf r_0)/t_2-\mathbf w\) 的单位向量定义早期飞行方向。角速度在该方向的投影采用较紧的零均值正态先验，法平面内采用各向同性的较宽先验。宽先验的平行和横向标准差分别为 10 和 35 rad/s，强先验分别为 5 和 20 rad/s。三轴分量均参与求解。先验方向由早期观测给出，沿飞行方向的自旋后验预计受该正则化显著影响。输出的观测灵敏度奇异值按先验尺度标准化，便于检查弱可辨识方向。

批量拟合保留高斯误差和精确舍入区间两种似然。MCMC 用白化的随机游走 Metropolis，仅在预热阶段调整步长。严格舍入另用区间内点和各区间面的局部曲率调整建议步幅，接受判定仍使用原始区间条件。

EKF 只使用高斯坐标误差，两次位置更新采用九维 \([\mathbf r,\mathbf v,\boldsymbol\omega]\) 状态、包含 \(dC_D/d|\mathbf u|\) 的动力学雅可比，以及 Joseph 形式协方差更新。该近似不施加 MCMC 的硬参数边界。合成验证使用早期两帧进行反演，将后续穿越点单独留作评分。

## 可选实测验证

```powershell
python issia.py --csv <ISSIA-3D.csv>
python dfl_j03woh.py --root <DFL-XML-directory>
```

ISSIA-3D 检查虚拟目标平面，DFL J03WOH 检查实际球门线。两者采样率为 25 Hz，使用飞行中的起点和 0.04、0.08 s 两帧，适合辅助检验短窗预测。[ISSIA-3D 数据](https://github.com/mguti97/SoccerNet-v3D/releases) · [DFL 数据](https://doi.org/10.6084/m9.figshare.28196177)

