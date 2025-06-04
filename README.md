# 基于ROOT框架实现对未知扰动斜抛运动的发射角预测系统 - ROOT C++实现

## 项目概述

本项目是基于ROOT框架的C++实现，用于进行基于ROOT框架实现对未知扰动斜抛运动的发射角预测。该系统通过多种数学模型和优化算法，对测量数据中的方位角(theta)进行校正，提高角度测量的精确度。

## 功能特点

- 多项式拟合分析
- 最优k值搜索与精细调整
- 解析曲面拟合
- 残差分析
- k函数优化
- theta扰动优化
- 分段校正函数优化

## 环境要求

- C++编译器(支持C++11或更高版本)
- ROOT框架(CERN开发的数据分析框架)
- Eigen库(线性代数计算)

## 安装与编译

1. 确保已安装ROOT环境，可从[ROOT官网](https://root.cern.ch/)下载安装
2. 确保已安装Eigen库
3. 编译方法:

  ```
root ROOT_oblique projection motion fitting.cpp
  ```

程序会自动执行以下步骤:
1. 多项式拟合
2. 大范围k值搜索
3. 解析曲面拟合
4. 精细k值搜索
5. 残差分析
6. k函数优化
7. theta扰动优化
8. theta校正函数优化

所有生成的图像将保存在`theta_picture`目录中。

## 工作原理

### 数据结构

程序使用`MeasurementData`结构体存储实验数据:
- r: 距离(米)
- alpha: 仰角(度)
- theta: 方位角(度)

### 主要算法流程

1. **多项式拟合**: 使用高阶多项式在r-alpha-theta空间中拟合数据点
2. **k值搜索**: 在指定范围内搜索最优k值，使理论计算的theta值与实验值的均方根误差(RMSE)最小
3. **解析曲面拟合**: 基于物理模型构建解析曲面
4. **残差分析**: 分析理论模型与实验数据之间的差异
5. **k函数优化**: 将k值扩展为r和alpha的函数，进一步优化模型
6. **theta扰动优化**: 引入扰动项修正theta值
7. **校正函数优化**: 构建分段校正函数，针对不同角度范围使用不同的校正模型

### 数学模型

- **k函数模型**:
  ```
  k(r, alpha) = k_base * (1 + a*r + b*alpha + c*r*alpha + d*r^2 + e*alpha^2)
  ```

- **扰动模型**:
  ```
  perturb(r, alpha) = a1 + a2*r + a3*sin(alpha) + a4*sin(2*alpha) + a5*r*sin(alpha)
  ```

- **校正函数**:
  程序使用分段校正函数，对低角度(≤45°)和高角度(>45°)分别采用不同维度的特征模型进行校正

## 输出结果

程序运行后会生成多个图像文件，保存在`theta_picture`目录中:

1. `step1_polynomial_fitting_error.png`: 多项式拟合残差分析
2. `step2_k_search.png`: 大范围k值搜索结果
3. `step3_analytical_surface.png`: 解析曲面拟合
4. `step4_fine_k_search.png`: 精细k值搜索结果
5. `step5_residual_analysis.png`: 残差分析图
6. `step6_k_function_optimization.png`: k函数优化结果
7. `step8_correction_function.png`: 校正函数优化结果

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请联系项目维护者。 