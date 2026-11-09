# 足球轨迹预测 · 三轴自旋版

从已知起点和两帧球心坐标估计 \(x=11\) m 平面的穿越位置和预测不确定度。默认是宽先验、各轴 1 cm 高斯观测误差和 4,000 个后验保留样本。每次针对单次射门拟合初始状态。

## 首轮运行（PowerShell）

在项目目录执行：

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -p "test_*.py" -v
python predict.py example_input.json --noise gaussian --sigma 0.01 --prior wide --draws 4000 --burn 1500 --output results/initial_4000
python predict.py example_input.json --method ekf --noise gaussian --sigma 0.01 --prior wide --draws 4000 --output results/initial_ekf_4000
```

MCMC 每条链预热 1,500 步，预热后每两步保留一个样本，每条链保留 1,000 个。高斯误差标准差 1 cm 是首轮假设。若确认坐标只经过厘米舍入，可显式使用 `--noise rounding --halfwidth 0.005`，并用单独的输出目录。舍入似然要求六个预测坐标都落在对应区间内。区间太窄或固定气动参数与观测不相容时，检查拟合错误和链诊断。

在 `summary.json` 中检查 `mode_yz`（穿越点的密度众数）、`region_95_area`（穿越条件下 95% 区域面积）、`p_cross` 和 `p_goal`。`p_goal` 表示球心落在题目给定球门矩形内的概率。MCMC 另检查 `diagnostics.reliable`，条件为全部 split-\( \hat R < 1.05 \) 且最小 ESS ≥ 200。EKF 从最后一帧的九维高斯近似抽样，样本保存在 `samples.npz` 的 `states` 中。MCMC 保存惯性系初速度和初始角速度，顺序见 `parameter_names`。`density.npz` 保存 95% 区域的密度网格。

复制 `example_input.json` 即可换案例。`r0` 和 `observations` 的单位是米，`times` 的单位是秒。\(x\) 朝向目标平面，\(z\) 从地面向上。计算球门概率时，\(y=0\) 对应球门横向中心。

## 初步验证和复现

```powershell
python validate.py --shots 3 --draws 4000 --output results/synthetic_initial
python run_analysis.py --deterministic-only
```

合成验证让九维 EKF 与批量拟合的 Laplace 近似共享早期观测，比较留出的真实穿越点、误差和 95% 区域覆盖。包含同模型、较高阻力、横风、较快自旋衰减四种条件。每组 3 球适合做流程检查。若要检查旧版冻结的 24 个合成案例对两种先验的敏感性，可执行 `python compare_rounding_priors.py --draws 4000 --workers 4`。这组数据由旧动力学生成，结果会记录改进模型的严格舍入不相容案例。

`python reproduce_example.py --workers 4` 运行四组观测模型和先验组合，每组保留 4,000 个后验样本。`python plot_results.py` 绘制四组示例等密度线。确定性分析会导出可供 `matlab/verify_cases.m` 检查的输入。

默认气动参数、来源及可辨识性说明见 [METHODS.md](METHODS.md)。

