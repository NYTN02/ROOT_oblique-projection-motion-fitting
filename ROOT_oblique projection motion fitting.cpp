/**
 * ROOT_oblique_projection_motion_fitting.cpp
 * 功能：theta角度校正系统的ROOT C++实现
 * 
 * 使用说明：
 * 1. 该程序需要在ROOT环境下编译和运行
 */

 #include <iostream>
 #include <vector>
 #include <cmath>
 #include <string>
 #include <fstream>
 #include <algorithm>
 #include <iomanip>
 #include <Eigen/Dense>
 
 // ROOT头文件（需要配置ROOT环境）
 #include "TROOT.h"
 #include "TFile.h"
 #include "TH1D.h"
 #include "TH2D.h"
 #include "TGraph.h"
 #include "TGraph2D.h"
 #include "TCanvas.h"
 #include "TF1.h"
 #include "TF2.h"
 #include "TStyle.h"
 #include "TLegend.h"
 #include "TText.h"
 #include "TMath.h"
 #include "TMatrixD.h"
 #include "TVectorD.h"
 #include "TFitResult.h"
 #include "TFitResultPtr.h"
 #include "TMultiGraph.h"
 #include "TColor.h"
 #include "TPaveText.h"
 #include "TMinuitMinimizer.h"
 #include "Math/Functor.h"
 #include "TLatex.h"
 #include "Math/Minimizer.h"
 #include "Math/Factory.h"
 
 // 数据结构
 struct MeasurementData {
     std::vector<double> r;     // 距离（米）
     std::vector<double> alpha; // 仰角（度）
     std::vector<double> theta; // 方位角（度）
 };
 
 // k函数模型
 auto k_model = [](double r, double alpha, const double* p) {
     return p[0] * (1 + p[1]*r + p[2]*alpha + p[3]*r*alpha + p[4]*r*r + p[5]*alpha*alpha);
 };
 
 // 全局变量，用于存储优化后的扰动参数
 double g_perturb_params[5] = {0, 0, 0, 0, 0};
 
 // 函数声明
 void loadData(MeasurementData &data);
 void polynomialFitting(const MeasurementData &data);
 double kSearch(const MeasurementData &data, double k_min, double k_max);
 void analyticalSurface(const MeasurementData &data, double k_opt);
 double calculateRMSE(double k, const MeasurementData &data);
 void calculateTheta(double k, const MeasurementData &data, std::vector<double> &theta_calc, std::vector<bool> &valid_points);
 void fineTuneK(const MeasurementData &data, double k_initial);
 void residualAnalysis(const MeasurementData &data, double k_opt_final);
 void kFunctionOptimization(const MeasurementData &data, double k_base);
 void thetaPerturbationOptimization(const MeasurementData &data, double k_base, const double* k_params);
 void thetaCorrectionFunctionOptimization(const MeasurementData &data, double k_base, const double* k_params, const double* perturb_params);
 
 /**
  * 加载实验数据
  */
 void loadData(MeasurementData &data) {
     // 清空数据
     data.r.clear();
     data.alpha.clear();
     data.theta.clear();
     
     // 硬编码数据
     // 格式：[r, alpha, theta]
     double raw_data[][3] = {
         {148, -7.35, 0.12},
         {139, -0.56, 5.86},
         {80, 0.27, 10.62},
         {235, 0.5, 15.27},
         {287, -0.35, 20.13},
         {253, 7.43, 25.23},
         {245, 12.63, 30.65},
         {287, 10.84, 35.45},
         {258, 17.57, 40.06},
         {308, 10.84, 45.02},
         {205, 33.46, 54.8},
         {208, 33.93, 58.88},
         {224, 23.34, 62.6},
         {219, 23.66, 63.36},
         {215, 17.99, 65.75},
         {214, 15.74, 66.32},
         {209, 13.01, 67.4},
         {204, 10.42, 68.38},
         {196, 8.48, 69.55},
         {184, 9.44, 70.83},
         {183, 1.44, 71.53},
         {166, 13.75, 72.49},
         {161, 12.91, 73.15},
         {155, 0.8, 74.45},
         {144, 3.8, 75.46},
         {136, -4.91, 76.5},
         {126, -7.44, 77.57},
         {115, -8.3, 78.59},
         {104, -13.32, 79.73},
         {96, -15.1, 80.5}
     };
     
     // 读取数据到向量
     int num_points = sizeof(raw_data) / sizeof(raw_data[0]);
     for (int i = 0; i < num_points; i++) {
         data.r.push_back(raw_data[i][0]);
         data.alpha.push_back(raw_data[i][1]);
         data.theta.push_back(raw_data[i][2]);
     }
     
     std::cout << "已加载 " << num_points << " 条实验数据" << std::endl;
 }
 
 /**
  * 多项式拟合
  */
 void polynomialFitting(const MeasurementData &data) {
     // 创建TGraph2D对象用于r-alpha-theta空间中的拟合
     TGraph2D *g = new TGraph2D();
     g->SetTitle("3D Data Points in r-alpha-theta Space");
     g->SetMarkerStyle(20);
     g->SetMarkerColor(kGreen);
     g->SetMarkerSize(2.0);
     
     // 填充数据点
     for (size_t i = 0; i < data.r.size(); i++) {
         g->SetPoint(i, data.r[i], data.alpha[i], data.theta[i]);
     }
     
     // 创建多项式拟合函数 (在r-alpha空间拟合theta)
     // theta = p00 + p10*r + p01*alpha + p20*r^2 + p11*r*alpha + p02*alpha^2 + 
     //         p30*r^3 + p21*r^2*alpha + p12*r*alpha^2 + p03*alpha^3
     TF2 *f2 = new TF2("f2", "[0] + [1]*x + [2]*y + [3]*x*x + [4]*x*y + [5]*y*y + [6]*x*x*x + [7]*x*x*y + [8]*x*y*y + [9]*y*y*y", 
                      *std::min_element(data.r.begin(), data.r.end()), *std::max_element(data.r.begin(), data.r.end()),
                      *std::min_element(data.alpha.begin(), data.alpha.end()), *std::max_element(data.alpha.begin(), data.alpha.end()));
     
     // 设置初始参数
     for (int i = 0; i < 10; i++) {
         f2->SetParameter(i, 0.0);
     }
     
     // 进行拟合
     TFitResultPtr fitResult = g->Fit(f2, "S");
     
     // 创建画布
     TCanvas *c1 = new TCanvas("c1", "多项式拟合结果", 1600, 1200);
     c1->SetRightMargin(0.15);
     c1->SetFillColor(0);
     c1->SetFrameFillColor(0);
     
     // 设置画布属性
     c1->SetHighLightColor(2);
     c1->SetFillStyle(4000);  // 透明背景
     c1->SetFrameFillStyle(4000);
     gStyle->SetOptFit(1);
     
     // 绘制拟合曲面
     f2->SetNpx(100);
     f2->SetNpy(100);
     f2->SetTitle("3D Data Points and Polynomial Fitting Surface in r-alpha-theta Space");
     f2->GetXaxis()->SetTitle("Distance r (m)");
     f2->GetYaxis()->SetTitle("Elevation Angle alpha (deg)");
     f2->GetZaxis()->SetTitle("Azimuth Angle theta (deg)");
     f2->Draw("surf1");
     
     // 添加原始数据点
     g->Draw("same p");
     
     // 创建图例
     TLegend *legend = new TLegend(0.65, 0.75, 0.85, 0.85);
     legend->AddEntry(f2, "Fitting Surface", "l");
     legend->AddEntry(g, "Data Points", "p");
     legend->Draw();
     
     // 设置视角
     c1->SetTheta(45);
     c1->SetPhi(30);
     
     // 输出拟合统计信息
     double chi2 = fitResult->Chi2();
     int ndf = fitResult->Ndf();
     double r2 = 1.0 - chi2 / ndf; // 近似计算R^2
 
     std::cout << "Fitting Goodness R-square: " << r2 << std::endl;
     std::cout << "Chi2/ndf: " << chi2 / ndf << std::endl;
     
     // 输出系数信息
     std::cout << "\nPolynomial Coefficients:" << std::endl;
     const char* coeff_names[] = {"Constant", "r", "alpha", "r^2", "r*alpha", "alpha^2", "r^3", "r^2*alpha", "r*alpha^2", "alpha^3"};
     for (int i = 0; i < 10; i++) {
         std::cout << coeff_names[i] << ": " << f2->GetParameter(i) << std::endl;
     }
     
     // 计算拟合残差
     std::vector<double> theta_fit(data.r.size());
     for (size_t i = 0; i < data.r.size(); i++) {
         theta_fit[i] = f2->Eval(data.r[i], data.alpha[i]);
     }
     std::vector<double> residuals(data.r.size());
     for (size_t i = 0; i < data.r.size(); i++) {
         residuals[i] = data.theta[i] - theta_fit[i];
     }
     
     // 计算RMSE
     double sum_squared_error = 0.0;
     for (double res : residuals) {
         sum_squared_error += res * res;
     }
     double rmse = sqrt(sum_squared_error / residuals.size());
     std::cout << "多项式拟合RMSE: " << rmse << " 度" << std::endl;
     
     // 创建误差直方图
     gStyle->SetOptStat(0);  // 禁用默认统计框
     gStyle->SetOptFit(0);   // 禁用默认拟合框
     
     TCanvas *c1_err = new TCanvas("c1_err", "Residuals of Polynomial Fitting", 1600, 1200);
     c1_err->SetRightMargin(0.15);
     c1_err->SetFillColor(0);
     c1_err->SetFrameFillColor(0);
     
     // 设置画布属性
     c1_err->SetHighLightColor(2);
     c1_err->SetFillStyle(4000);
     c1_err->SetFrameFillStyle(4000);
     
     TH1D *h_res = new TH1D("h_res", "Residuals of Polynomial Fitting;theta_{exp} - theta_{fit} (deg);Counts", 20, -(*std::max_element(residuals.begin(), residuals.end())), *std::max_element(residuals.begin(), residuals.end()));
     for (size_t i = 0; i < residuals.size(); i++) {
         h_res->Fill(residuals[i]);
     }
     h_res->SetFillColorAlpha(kBlue, 0.35);
     h_res->SetFillStyle(1001);
     h_res->SetLineColor(kBlue+2);
     h_res->SetLineWidth(2);
     h_res->SetBarWidth(0.8);
     h_res->SetBarOffset(0.1);
     h_res->Draw("B");  // 使用"B"选项绘制带间隔的柱状图
     
     // 拟合正态分布
     TF1 *f_gaus = new TF1("f_gaus", "gaus", -(*std::max_element(residuals.begin(), residuals.end())), *std::max_element(residuals.begin(), residuals.end()));
     f_gaus->SetLineColor(kRed);
     f_gaus->SetLineWidth(2);
     h_res->Fit(f_gaus, "Q");  // 使用安静模式拟合
     f_gaus->Draw("same");
     
     // 添加统计信息框
     TPaveText *stats = new TPaveText(0.65, 0.65, 0.85, 0.85, "NDC");
     stats->SetFillColor(0);
     stats->SetBorderSize(0);
     stats->SetTextAlign(12);
     stats->SetTextSize(0.035);
     char text[100];
     sprintf(text, "#chi^{2} / ndf = %.3f / %d", f_gaus->GetChisquare(), f_gaus->GetNDF());
     stats->AddText(text);
     sprintf(text, "Constant = %.3f #pm %.3f", f_gaus->GetParameter(0), f_gaus->GetParError(0));
     stats->AddText(text);
     sprintf(text, "Mean = %.3f #pm %.3f", f_gaus->GetParameter(1), f_gaus->GetParError(1));
     stats->AddText(text);
     sprintf(text, "Sigma = %.3f #pm %.3f", f_gaus->GetParameter(2), f_gaus->GetParError(2));
     stats->AddText(text);
     stats->Draw();
     
     // 添加图例
     TLegend *leg_hist = new TLegend(0.15, 0.75, 0.45, 0.85);
     leg_hist->SetBorderSize(0);
     leg_hist->SetFillStyle(0);
     leg_hist->SetTextSize(0.045);
     leg_hist->AddEntry(h_res, "Residuals", "f");
     leg_hist->AddEntry(f_gaus, "Gaussian Fit", "l");
     leg_hist->Draw();
     
     // 保存图像
     c1_err->Print("theta_picture/step1_polynomial_fitting_error.png", "png");
 }
 
 /**
  * 计算theta角度
  * @param k 校正系数
  * @param data 测量数据
  * @param theta_calc 输出：计算得到的theta值
  * @param valid_points 输出：表示哪些点是有效的（有物理解）
  */
 void calculateTheta(double k, const MeasurementData &data, std::vector<double> &theta_calc, std::vector<bool> &valid_points) {
     // 确保输出向量大小正确
     theta_calc.resize(data.r.size());
     valid_points.resize(data.r.size());
     
     // 对每个点进行计算
     for (size_t i = 0; i < data.r.size(); i++) {
         // 转换为弧度
         double alpha_rad = data.alpha[i] * TMath::Pi() / 180.0;
         double cos_alpha = TMath::Cos(alpha_rad);
         double sin_alpha = TMath::Sin(alpha_rad);
         double kr = k * data.r[i];
         
         // 计算判别式
         double discriminant = 1.0 - kr * (2.0 * sin_alpha + kr * cos_alpha * cos_alpha);
         
         // 检查是否有物理解
         if (discriminant >= 0.0 && TMath::Abs(cos_alpha) > 1e-10) {
             double sqrt_term = TMath::Sqrt(discriminant);
             double denom = kr * cos_alpha;
             
             // 计算两个可能的解
             double theta1 = TMath::ATan2(1.0 + sqrt_term, denom) * 180.0 / TMath::Pi();
             double theta2 = TMath::ATan2(1.0 - sqrt_term, denom) * 180.0 / TMath::Pi();
             
             // 规范化角度到[0, 90]范围
             theta1 = fmod(theta1 + 360.0, 360.0);
             theta2 = fmod(theta2 + 360.0, 360.0);
             
             if (theta1 > 90.0 && theta1 <= 270.0) {
                 theta1 = 180.0 - theta1;
             } else if (theta1 > 270.0) {
                 theta1 = theta1 - 360.0;
             }
             
             if (theta2 > 90.0 && theta2 <= 270.0) {
                 theta2 = 180.0 - theta2;
             } else if (theta2 > 270.0) {
                 theta2 = theta2 - 360.0;
             }
             
             // 选择更接近测量值的解
             if (TMath::Abs(theta1 - data.theta[i]) <= TMath::Abs(theta2 - data.theta[i])) {
                 theta_calc[i] = theta1;
             } else {
                 theta_calc[i] = theta2;
             }
             
             valid_points[i] = true;
         } else {
             theta_calc[i] = 0.0;  // 无效值
             valid_points[i] = false;
         }
     }
 }
 
 /**
  * 计算均方根误差
  * @param k 校正系数
  * @param data 测量数据
  * @return RMSE值
  */
 double calculateRMSE(double k, const MeasurementData &data) {
     std::vector<double> theta_calc;
     std::vector<bool> valid_points;
     
     // 计算theta值
     calculateTheta(k, data, theta_calc, valid_points);
     
     // 计算均方根误差
     double sum_squared_error = 0.0;
     int valid_count = 0;
     
     for (size_t i = 0; i < data.r.size(); i++) {
         if (valid_points[i]) {
             double error = theta_calc[i] - data.theta[i];
             sum_squared_error += error * error;
             valid_count++;
         }
     }
     
     // 如果没有有效点，返回极大值
     if (valid_count == 0) {
         return 1e6;
     }
     
     return TMath::Sqrt(sum_squared_error / valid_count);
 }
 
 /**
  * 大范围搜索k值
  * @return 返回最优k值
  */
 double kSearch(const MeasurementData &data, double k_min, double k_max) {
     // 搜索网格点数
     const int num_points = 100;
     std::vector<double> k_values(num_points);
     std::vector<double> rmse_values(num_points);
     
     // 在搜索范围内均匀分布k值
     double step = (k_max - k_min) / (num_points - 1);
     for (int i = 0; i < num_points; i++) {
         k_values[i] = k_min + i * step;
     }
     
     // 计算每个k值对应的RMSE
     double best_rmse = 1e6;
     double k_opt = k_min;
     
     for (int i = 0; i < num_points; i++) {
         rmse_values[i] = calculateRMSE(k_values[i], data);
         
         if (rmse_values[i] < best_rmse) {
             best_rmse = rmse_values[i];
             k_opt = k_values[i];
         }
     }
     
     // 创建画布
     TCanvas *c2 = new TCanvas("c2", "大范围k值搜索结果", 1600, 1200);
     c2->SetRightMargin(0.15);
     c2->SetFillColor(0);
     c2->SetFrameFillColor(0);
     c2->SetHighLightColor(2);
     c2->SetFillStyle(4000);
     c2->SetFrameFillStyle(4000);
     
     // 设置画布属性
     c2->SetFillColor(0);
     c2->SetFrameFillColor(0);
     gStyle->SetCanvasPreferGL(true);
     
     // 创建图形
     TGraph *g = new TGraph(num_points, &k_values[0], &rmse_values[0]);
     g->SetTitle("RMSE vs k Value (Large Range Search)");
     g->GetXaxis()->SetTitle("k Value");
     g->GetYaxis()->SetTitle("RMSE (degrees)");
     g->SetLineColor(kBlue);
     g->SetLineWidth(2);
     g->Draw("AL"); // A: 轴，L: 线
     
     // 标记最优点
     TGraph *g_opt = new TGraph(1, &k_opt, &best_rmse);
     g_opt->SetMarkerStyle(20);
     g_opt->SetMarkerColor(kRed);
     g_opt->SetMarkerSize(1.5);
     g_opt->Draw("P"); // P: 点
     
     // 添加文字说明
     char text[100];
     sprintf(text, "k = %.6f\nRMSE = %.4f degrees", k_opt, best_rmse);
     
     TPaveText *pt = new TPaveText(k_opt + 0.0005, best_rmse, k_opt + 0.003, best_rmse + 0.5, "NB");
     pt->AddText(text);
     pt->SetFillColor(0);
     pt->Draw();
     
     // 添加网格
     c2->SetGrid();
     
     // 保存前删除画布
     c2->SaveAs("theta_picture/step2_k_search.png", "png");
     
     // 输出结果
     std::cout << "Large Range Search Results:" << std::endl;
     std::cout << "Initial Optimal k Value: " << k_opt << std::endl;
     std::cout << "Initial Minimum RMSE: " << best_rmse << " degrees" << std::endl;
     
     // 增大字体
     g->GetXaxis()->SetTitleSize(0.05);
     g->GetXaxis()->SetLabelSize(0.045);
     g->GetYaxis()->SetTitleSize(0.05);
     g->GetYaxis()->SetLabelSize(0.045);
     
     // 图例增大字体
     TLegend *legend = new TLegend(0.65, 0.75, 0.85, 0.85);
     legend->SetTextSize(0.04);
     
     // 标注增大字体
     pt->SetTextSize(0.04);
     pt->SetTextAlign(12);
     pt->SetBorderSize(0);
     
     return k_opt;
 }
 
 /**
  * 解析式拟合曲面
  */
 void analyticalSurface(const MeasurementData &data, double k_opt) {
     std::cout << "Using k = " << k_opt << std::endl;
     
     // 创建画布
     TCanvas *c3 = new TCanvas("c3", "Analytical Surface", 1600, 1200);
     c3->SetRightMargin(0.15);
     
     // 设置3D图形参数
     gStyle->SetPalette(kRainBow);
     c3->SetFillColor(0);
     c3->SetFrameFillColor(0);
     c3->SetHighLightColor(2);
     c3->SetFillStyle(4000);
     c3->SetFrameFillStyle(4000);
     
     // 启用OpenGL渲染
     gStyle->SetCanvasPreferGL(true);
     
     // 创建密集网格
     const int n_points = 1000;  // 进一步增加网格密度
     double alpha_min = -90.0;
     double alpha_max = 90.0;
     double r_min = 0.0;
     double r_max = 400.0;
     
     // 创建TGraph2D对象
     TGraph2D* g_pos = new TGraph2D();
     TGraph2D* g_neg = new TGraph2D();
     int idx_pos = 0, idx_neg = 0;
     
     // 为每个点计算theta
     for (int i = 0; i < n_points; i++) {
         double alpha = alpha_min + (alpha_max - alpha_min) * i / (n_points - 1);
         double alpha_rad = alpha * TMath::Pi() / 180.0;
         double cos_alpha = TMath::Cos(alpha_rad);
         double sin_alpha = TMath::Sin(alpha_rad);
         
         for (int j = 0; j < n_points; j++) {
             double r = r_min + (r_max - r_min) * j / (n_points - 1);
             double kr = k_opt * r;
             
             // 计算判别式
             double discriminant = 1.0 - kr * (2.0 * sin_alpha + kr * cos_alpha * cos_alpha);
             
             // 只对有效点计算theta
             if (discriminant >= 0.0 && TMath::Abs(cos_alpha) > 1e-10) {
                 double sqrt_term = TMath::Sqrt(discriminant);
                 double denom = kr * cos_alpha;
                 
                 // 计算两个可能的解
                 double theta1 = TMath::ATan2(1.0 + sqrt_term, denom) * 180.0 / TMath::Pi();
                 double theta2 = TMath::ATan2(1.0 - sqrt_term, denom) * 180.0 / TMath::Pi();
                 
                 // 规范化角度
                 theta1 = fmod(theta1 + 360.0, 360.0);
                 theta2 = fmod(theta2 + 360.0, 360.0);
                 
                 if (theta1 > 90.0 && theta1 <= 270.0) {
                     theta1 = 180.0 - theta1;
                 } else if (theta1 > 270.0) {
                     theta1 = theta1 - 360.0;
                 }
                 
                 if (theta2 > 90.0 && theta2 <= 270.0) {
                     theta2 = 180.0 - theta2;
                 } else if (theta2 > 270.0) {
                     theta2 = theta2 - 360.0;
                 }
                 
                 // 只添加有效的theta值（0-90度范围内）
                 if (theta1 >= 0.0 && theta1 <= 90.0) {
                     g_pos->SetPoint(idx_pos++, alpha, r, theta1);
                 }
                 if (theta2 >= 0.0 && theta2 <= 90.0) {
                     g_neg->SetPoint(idx_neg++, alpha, r, theta2);
                 }
             }
         }
     }
     
     // 设置点的样式和颜色
     g_pos->SetMarkerStyle(20);
     g_pos->SetMarkerColor(kRed);
     g_pos->SetMarkerSize(0.5);
     g_pos->SetLineColor(kRed);
     g_pos->SetLineWidth(1);
     g_pos->SetFillColor(kRed);
     g_pos->SetFillStyle(1001);
     
     g_neg->SetMarkerStyle(20);
     g_neg->SetMarkerColor(kBlue);
     g_neg->SetMarkerSize(0.5);
     g_neg->SetLineColor(kBlue);
     g_neg->SetLineWidth(1);
     g_neg->SetFillColor(kBlue);
     g_neg->SetFillStyle(1001);
     
     // 设置绘图选项
     g_pos->SetTitle("Analytical Surface");
     g_pos->GetXaxis()->SetTitle("#alpha (deg)");
     g_pos->GetYaxis()->SetTitle("r (m)");
     g_pos->GetZaxis()->SetTitle("#theta (deg)");
     
     // 设置轴标签大小
     g_pos->GetXaxis()->SetTitleSize(0.035);
     g_pos->GetYaxis()->SetTitleSize(0.035);
     g_pos->GetZaxis()->SetTitleSize(0.035);
     g_pos->GetXaxis()->SetLabelSize(0.03);
     g_pos->GetYaxis()->SetLabelSize(0.03);
     g_pos->GetZaxis()->SetLabelSize(0.03);
     
     // 创建实验数据点
     TGraph2D *g_data = new TGraph2D(data.alpha.size());
     g_data->SetMarkerStyle(20);
     g_data->SetMarkerColor(kGreen+2);
     g_data->SetMarkerSize(1.5);
     g_data->SetLineColor(kGreen+2);
     g_data->SetLineWidth(2);
     
     for (size_t i = 0; i < data.alpha.size(); i++) {
         g_data->SetPoint(i, data.alpha[i], data.r[i], data.theta[i]);
     }
     
     // 设置3D视图
     c3->cd();
     TPad *pad = new TPad("pad", "pad", 0, 0, 1, 1);
     pad->SetFillColor(0);
     pad->Draw();
     pad->cd();
     
     // 绘制图形
     g_pos->Draw("surf1");
     g_neg->Draw("surf1 same");
     g_data->Draw("P same");
     
     // 设置视角
     pad->SetTheta(130);
     pad->SetPhi(60);
     
     // 添加图例
     TLegend *legend = new TLegend(0.65, 0.75, 0.85, 0.90);
     legend->SetTextSize(0.03);
     legend->SetBorderSize(1);
     legend->SetLineWidth(2);
     legend->SetFillStyle(0);
     legend->AddEntry(g_pos, "Positive Root", "lf");
     legend->AddEntry(g_neg, "Negative Root", "lf");
     legend->AddEntry(g_data, "Experimental Data", "p");
     legend->Draw();
     
     // 保存图像（使用高质量设置）
     c3->Print("theta_picture/step3_analytical_surface.png", "png");
 }
 
 /**
  * 精细k值搜索
  */
 void fineTuneK(const MeasurementData &data, double k_initial) {
     // 在k_initial附近进行精细搜索
     double k_center = k_initial;
     double k_range = 0.0002;  // 搜索范围缩小到±0.0001
     const int num_points = 300;  // 增加搜索点数以提高精度
     
     // 创建精细的k值网格
     std::vector<double> k_fine(num_points);
     std::vector<double> rmse_fine(num_points);
     
     // 生成k值序列
     for (int i = 0; i < num_points; i++) {
         k_fine[i] = k_center - k_range/2 + k_range * i / (num_points - 1);
     }
     
     // 计算每个k值对应的RMSE
     std::cout << "Starting fine search for k, range: [" << k_fine[0] << ", " << k_fine[num_points-1] << "]" << std::endl;
     
     double best_rmse = 1e6;
     double k_opt_final = k_center;
     
     for (int i = 0; i < num_points; i++) {
         rmse_fine[i] = calculateRMSE(k_fine[i], data);
         if (rmse_fine[i] < best_rmse) {
             best_rmse = rmse_fine[i];
             k_opt_final = k_fine[i];
         }
     }
     
     // 输出结果
     std::cout << "\nFine Search Results:" << std::endl;
     std::cout << "Final Optimal k Value: " << k_opt_final << std::endl;
     std::cout << "Final Minimum RMSE: " << best_rmse << " degrees" << std::endl;
     std::cout << "Improvement from Initial k: " 
               << (calculateRMSE(k_initial, data) - best_rmse) / calculateRMSE(k_initial, data) * 100 
               << "%" << std::endl;
     
     // 创建画布
     TCanvas *c4 = new TCanvas("c4", "Fine k Value Search Results", 1600, 1200);
     c4->SetRightMargin(0.15);
     c4->SetFillColor(0);
     c4->SetFrameFillColor(0);
     c4->SetHighLightColor(2);
     c4->SetFillStyle(4000);
     c4->SetFrameFillStyle(4000);
     
     // 设置画布属性
     c4->SetFillColor(0);
     c4->SetFrameFillColor(0);
     gStyle->SetCanvasPreferGL(true);
     
     // 创建图形
     TMultiGraph *mg = new TMultiGraph();
     
     // 创建RMSE曲线
     TGraph *g_rmse = new TGraph(num_points);
     for (int i = 0; i < num_points; i++) {
         g_rmse->SetPoint(i, k_fine[i], rmse_fine[i]);
     }
     g_rmse->SetLineColor(kBlue);
     g_rmse->SetLineWidth(2);
     mg->Add(g_rmse, "L");
     
     // 创建最优点标记
     TGraph *g_opt = new TGraph(1);
     g_opt->SetPoint(0, k_opt_final, best_rmse);
     g_opt->SetMarkerStyle(20);
     g_opt->SetMarkerColor(kRed);
     g_opt->SetMarkerSize(2.0);  // 增大标记点大小
     mg->Add(g_opt, "P");
     
     // 绘制图形
     mg->Draw("A");
     
     // 设置标题和轴标签的字体大小
     mg->GetXaxis()->SetTitleSize(0.05);  // 增大X轴标题
     mg->GetXaxis()->SetLabelSize(0.045);  // 增大X轴刻度标签
     mg->GetYaxis()->SetTitleSize(0.05);  // 增大Y轴标题
     mg->GetYaxis()->SetLabelSize(0.045);  // 增大Y轴刻度标签
     mg->GetXaxis()->SetTitleOffset(1.2);  // 调整标题位置
     mg->GetYaxis()->SetTitleOffset(1.2);
     
     // 设置主标题
     TLatex *title = new TLatex();
     title->SetNDC();
     title->SetTextSize(0.05);
     title->SetTextAlign(23);
     title->DrawLatex(0.5, 0.95, "RMSE vs k Value (Fine Search)");
     
     mg->GetXaxis()->SetTitle("k Value");
     mg->GetYaxis()->SetTitle("RMSE (degrees)");
     
     // 设置坐标轴范围
     double k_margin = k_range/10;
     mg->GetXaxis()->SetLimits(k_center - k_range/2 - k_margin, k_center + k_range/2 + k_margin);
     
     // 添加图例
     TLegend *legend = new TLegend(0.65, 0.75, 0.85, 0.85);
     legend->SetTextSize(0.04);  // 增大图例文字
     legend->AddEntry(g_rmse, "RMSE Curve", "l");
     legend->AddEntry(g_opt, "Optimal Point", "p");
     legend->Draw();
     
     // 添加文字说明
     TPaveText *pt = new TPaveText(0.15, 0.75, 0.45, 0.85, "NDC");  // 使用NDC坐标系统
     pt->SetTextSize(0.04);  // 增大文字大小
     pt->SetTextAlign(12);  // 左对齐
     pt->AddText(Form("k = %.6f", k_opt_final));
     pt->AddText(Form("RMSE = %.4f deg", best_rmse));
     pt->SetFillColor(0);
     pt->SetBorderSize(0);
     pt->Draw();
     
     // 添加网格
     c4->SetGrid();
     
     // 保存前删除画布
     c4->SaveAs("theta_picture/step4_fine_k_search.png", "png");
     
     // 使用最终的k值重新生成解析曲面
     analyticalSurface(data, k_opt_final);
 }
 
 // 残差分析函数实现
 void residualAnalysis(const MeasurementData &data, double k_opt_final) {
     // 1. 计算理论theta和有效点
     std::vector<double> theta_theoretical, residuals, valid_r, valid_alpha, valid_theta, valid_residuals;
     std::vector<bool> valid_points;
     calculateTheta(k_opt_final, data, theta_theoretical, valid_points);
     // 2. 计算残差
     for (size_t i = 0; i < data.r.size(); ++i) {
         if (valid_points[i]) {
             double res = data.theta[i] - theta_theoretical[i];
             residuals.push_back(res);
             valid_r.push_back(data.r[i]);
             valid_alpha.push_back(data.alpha[i]);
             valid_theta.push_back(data.theta[i]);
             valid_residuals.push_back(res);
         }
     }
     // 3. 统计信息
     double sum = 0, sum2 = 0, max_res = -1e9, min_res = 1e9;
     for (double v : valid_residuals) {
         sum += v;
         sum2 += v * v;
         if (v > max_res) max_res = v;
         if (v < min_res) min_res = v;
     }
     double mean = sum / valid_residuals.size();
     double stddev = sqrt(sum2 / valid_residuals.size() - mean * mean);
     std::cout << "残差统计：" << std::endl;
     std::cout << "均值: " << mean << "°\n标准差: " << stddev << "°\n最大残差: " << max_res << "°\n最小残差: " << min_res << "°" << std::endl;
 
     // 4. 创建画布
     TCanvas *c5 = new TCanvas("c5", "Residual Analysis", 1600, 1200);
     c5->SetRightMargin(0.15);
     c5->SetFillColor(0);
     c5->SetFrameFillColor(0);
     c5->SetHighLightColor(2);
     c5->SetFillStyle(4000);
     c5->SetFrameFillStyle(4000);
     
     c5->Divide(2,2);
 
     // 4.1 残差-距离
     c5->cd(1);
     TGraph *g1 = new TGraph(valid_r.size(), &valid_r[0], &valid_residuals[0]);
     g1->SetTitle("Residual vs Distance;Distance r (m);Residual (deg)");
     g1->SetMarkerStyle(20);
     g1->SetMarkerColor(kBlue+2);
     g1->SetMarkerSize(1.2);
     g1->Draw("AP");
     TLine *l1 = new TLine(*std::min_element(valid_r.begin(), valid_r.end()), 0, *std::max_element(valid_r.begin(), valid_r.end()), 0);
     l1->SetLineStyle(2); l1->Draw();
 
     // 4.2 残差-仰角
     c5->cd(2);
     TGraph *g2 = new TGraph(valid_alpha.size(), &valid_alpha[0], &valid_residuals[0]);
     g2->SetTitle("Residual vs Alpha;Alpha (deg);Residual (deg)");
     g2->SetMarkerStyle(20);
     g2->SetMarkerColor(kMagenta+2);
     g2->SetMarkerSize(1.2);
     g2->Draw("AP");
     TLine *l2 = new TLine(*std::min_element(valid_alpha.begin(), valid_alpha.end()), 0, *std::max_element(valid_alpha.begin(), valid_alpha.end()), 0);
     l2->SetLineStyle(2); l2->Draw();
 
     // 4.3 残差直方图+正态分布
     c5->cd(3);
     int nbins = 20;  // 增加bin数量
     double min_hist = *std::min_element(valid_residuals.begin(), valid_residuals.end());
     double max_hist = *std::max_element(valid_residuals.begin(), valid_residuals.end());
     TH1D *h_res = new TH1D("h_res_hist", "Residual Distribution;Residual (deg);Probability", nbins, min_hist, max_hist);
     
     // 设置直方图样式
     h_res->SetFillColorAlpha(kBlue, 0.35);  // 设置半透明填充
     h_res->SetFillStyle(1001);
     h_res->SetLineColor(kBlue+2);
     h_res->SetLineWidth(2);
     h_res->SetBarWidth(0.8);    // 设置柱状图宽度
     h_res->SetBarOffset(0.1);   // 设置柱状图偏移
     
     // 填充数据
     for (double v : valid_residuals) {
         h_res->Fill(v);
     }
     h_res->Scale(1.0/h_res->GetEntries());
     
     // 设置Y轴范围，留出一定空间显示正态分布曲线
     double maxBinContent = h_res->GetMaximum();
     h_res->GetYaxis()->SetRangeUser(0, maxBinContent * 1.2);
     
     // 绘制直方图
     h_res->Draw("B");  // 使用"B"选项绘制带间隔的柱状图
     
     // 拟合正态分布
     TF1 *f_gaus = new TF1("f_gaus", "gaus", min_hist, max_hist);
     f_gaus->SetLineColor(kRed);
     f_gaus->SetLineWidth(2);
     h_res->Fit(f_gaus, "Q");
     f_gaus->Draw("same");
     
     // 添加统计信息框
     TPaveText *stats = new TPaveText(0.65, 0.65, 0.85, 0.85, "NDC");
     stats->SetFillColor(0);
     stats->SetBorderSize(0);
     stats->SetTextAlign(12);
     stats->SetTextSize(0.035);
     char text[100];
     sprintf(text, "#chi^{2} / ndf = %.3f / %d", f_gaus->GetChisquare(), f_gaus->GetNDF());
     stats->AddText(text);
     sprintf(text, "Constant = %.3f #pm %.3f", f_gaus->GetParameter(0), f_gaus->GetParError(0));
     stats->AddText(text);
     sprintf(text, "Mean = %.3f #pm %.3f", f_gaus->GetParameter(1), f_gaus->GetParError(1));
     stats->AddText(text);
     sprintf(text, "Sigma = %.3f #pm %.3f", f_gaus->GetParameter(2), f_gaus->GetParError(2));
     stats->AddText(text);
     stats->Draw();
     
     // 添加图例
     TLegend *leg_hist = new TLegend(0.15, 0.75, 0.45, 0.85);  // 调整位置和大小
     leg_hist->SetBorderSize(0);
     leg_hist->SetFillStyle(0);
     leg_hist->SetTextSize(0.045);  // 增大文字大小
     leg_hist->AddEntry(h_res, "Residuals", "f");
     leg_hist->AddEntry(f_gaus, "Gaussian Fit", "l");
     leg_hist->Draw();
     
     // 保存图像
     c5->SaveAs("theta_picture/step5_residual_analysis.png", "png");
     
     c5->cd(4);
     gStyle->SetCanvasPreferGL(true);  // 启用OpenGL渲染
     
     // 创建理论点和实验数据点的TGraph2D对象
     TGraph2D *g_theory = new TGraph2D(valid_points.size());
     TGraph2D *g_data = new TGraph2D(valid_points.size());
     
     int valid_count = 0;
     for (size_t i = 0; i < data.r.size(); ++i) {
         if (valid_points[i]) {
             g_theory->SetPoint(valid_count, data.alpha[i], data.r[i], theta_theoretical[i]);
             g_data->SetPoint(valid_count, data.alpha[i], data.r[i], data.theta[i]);
             valid_count++;
         }
     }
     
     // 设置理论点的样式
     g_theory->SetMarkerStyle(20);
     g_theory->SetMarkerColor(kBlue);
     g_theory->SetMarkerSize(0.8);
     g_theory->SetTitle("Theory vs Experiment");
     g_theory->GetXaxis()->SetTitle("Alpha (deg)");
     g_theory->GetYaxis()->SetTitle("r (m)");
     g_theory->GetZaxis()->SetTitle("Theta (deg)");
     
     // 设置实验点的样式
     g_data->SetMarkerStyle(24);
     g_data->SetMarkerColor(kRed);
     g_data->SetMarkerSize(1.2);
     
     // 绘制图形
     g_theory->Draw("P");
     g_data->Draw("P SAME");
     
     // 添加图例
     TLegend *legend = new TLegend(0.65, 0.75, 0.85, 0.85);
     legend->AddEntry(g_theory, "Theory", "p");
     legend->AddEntry(g_data, "Experiment", "p");
     legend->Draw();
     
     // 设置3D视图角度
     gPad->SetTheta(130);
     gPad->SetPhi(60);
 }
 
 // k函数优化函数实现
 void kFunctionOptimization(const MeasurementData &data, double k_base) {
     // 1. 定义k函数模型
     auto k_model = [](double r, double alpha, const double* p) {
         // p[0]=k_base, p[1]=a, p[2]=b, p[3]=c, p[4]=d, p[5]=e
         return p[0] * (1 + p[1]*r + p[2]*alpha + p[3]*r*alpha + p[4]*r*r + p[5]*alpha*alpha);
     };
     // 2. 定义RMSE目标函数
     auto rmse_func = [&](const double* p) {
         std::vector<double> theta_calc(data.r.size());
         std::vector<bool> valid_points(data.r.size());
         int valid_count = 0;
         double sum2 = 0;
         for (size_t i = 0; i < data.r.size(); ++i) {
             double k_val = k_model(data.r[i], data.alpha[i], p);
             std::vector<double> t_calc(1);
             std::vector<bool> v_pt(1);
             MeasurementData single;
             single.r = {data.r[i]};
             single.alpha = {data.alpha[i]};
             single.theta = {data.theta[i]};
             calculateTheta(k_val, single, t_calc, v_pt);
             if (v_pt[0]) {
                 double err = t_calc[0] - data.theta[i];
                 sum2 += err * err;
                 valid_count++;
             }
         }
         if (valid_count == 0) return 1e6;
         return sqrt(sum2 / valid_count);
     };
     // 3. 优化参数
     ROOT::Math::Minimizer* min = ROOT::Math::Factory::CreateMinimizer("Minuit2", "Simplex");
     ROOT::Math::Functor f(rmse_func, 6);
     min->SetFunction(f);
     min->SetLimitedVariable(0, "k_base", k_base, 1e-5, 0.00001, 0.01);
     min->SetVariable(1, "a", 0, 1e-5);
     min->SetVariable(2, "b", 0, 1e-5);
     min->SetVariable(3, "c", 0, 1e-7);
     min->SetVariable(4, "d", 0, 1e-8);
     min->SetVariable(5, "e", 0, 1e-8);
     min->Minimize();
     const double* p_opt = min->X();
     std::cout << "优化后参数：\n";
     std::cout << "k_base=" << p_opt[0] << ", a=" << p_opt[1] << ", b=" << p_opt[2] << ", c=" << p_opt[3] << ", d=" << p_opt[4] << ", e=" << p_opt[5] << std::endl;
     // 4. 计算优化后k和theta
     std::vector<double> k_values, theta_calc, valid_r, valid_alpha, valid_theta, valid_errors;
     std::vector<bool> valid_points;
     for (size_t i = 0; i < data.r.size(); ++i) {
         double k_val = k_model(data.r[i], data.alpha[i], p_opt);
         k_values.push_back(k_val);
         std::vector<double> t_calc(1);
         std::vector<bool> v_pt(1);
         MeasurementData single;
         single.r = {data.r[i]};
         single.alpha = {data.alpha[i]};
         single.theta = {data.theta[i]};
         calculateTheta(k_val, single, t_calc, v_pt);
         if (v_pt[0]) {
             theta_calc.push_back(t_calc[0]);
             valid_r.push_back(data.r[i]);
             valid_alpha.push_back(data.alpha[i]);
             valid_theta.push_back(data.theta[i]);
             valid_errors.push_back(t_calc[0] - data.theta[i]);
         }
     }
     // 5. 统计
     double rmse = 0, sum = 0, sum2 = 0, maxe = -1e9, mine = 1e9;
     for (double v : valid_errors) {
         sum += v;
         sum2 += v*v;
         if (v > maxe) maxe = v;
         if (v < mine) mine = v;
     }
     rmse = sqrt(sum2/valid_errors.size());
     double mean = sum/valid_errors.size();
     double stddev = sqrt(sum2/valid_errors.size() - mean*mean);
     std::cout << "优化后RMSE: " << rmse << "°\n均值: " << mean << "°\n标准差: " << stddev << "°\n最大误差: " << maxe << "°\n最小误差: " << mine << "°" << std::endl;
     // 6. 四合一图
     TCanvas *c6 = new TCanvas("c6", "k Function Optimization", 1600, 1200);
     c6->Divide(2,2);
     // 1. 测量vs计算
     c6->cd(1);
     TGraph *g1 = new TGraph(valid_theta.size(), &valid_theta[0], &theta_calc[0]);
     g1->SetTitle("Measured vs Calculated;Measured Theta (deg);Calculated Theta (deg)");
     g1->SetMarkerStyle(20);
     g1->SetMarkerColor(kBlue+2);
     g1->SetMarkerSize(1.2);
     g1->Draw("AP");
     TLine *l1 = new TLine(0,0,90,90); l1->SetLineStyle(2); l1->Draw();
     // 2. 残差分布
     c6->cd(2);
     int nbins = 15;
     double min_hist = *std::min_element(valid_errors.begin(), valid_errors.end());
     double max_hist = *std::max_element(valid_errors.begin(), valid_errors.end());
     TH1D *h_res = new TH1D("h_res_kopt", "Residual Distribution;Residual (deg);Probability", nbins, min_hist, max_hist);
     
     // 设置直方图样式
     h_res->SetFillColorAlpha(kBlue, 0.35);
     h_res->SetFillStyle(1001);
     h_res->SetLineColor(kBlue+2);
     h_res->SetLineWidth(2);
     h_res->SetBarWidth(0.8);
     h_res->SetBarOffset(0.1);
     
     // 填充数据
     for (double v : valid_errors) {
         h_res->Fill(v);
     }
     h_res->Scale(1.0/h_res->GetEntries());
     
     // 绘制直方图
     h_res->Draw("B");  // 使用"B"选项绘制带间隔的柱状图
     
     // 拟合高斯分布
     TF1 *f_gaus = new TF1("f_gaus_kopt", "gaus", min_hist, max_hist);
     f_gaus->SetLineColor(kRed);
     f_gaus->SetLineWidth(2);
     h_res->Fit(f_gaus, "Q");
     f_gaus->Draw("same");
     // 3. k与r
     c6->cd(3);
     TGraph *g2 = new TGraph(valid_r.size(), &valid_r[0], &k_values[0]);
     g2->SetTitle("k vs r;Distance r (m);k Value");
     g2->SetMarkerStyle(20);
     g2->SetMarkerColor(kGreen+2);
     g2->SetMarkerSize(1.2);
     g2->Draw("AP");
     // 4. k与alpha
     c6->cd(4);
     TGraph *g3 = new TGraph(valid_alpha.size(), &valid_alpha[0], &k_values[0]);
     g3->SetTitle("k vs Alpha;Alpha (deg);k Value");
     g3->SetMarkerStyle(20);
     g3->SetMarkerColor(kMagenta+2);
     g3->SetMarkerSize(1.2);
     g3->Draw("AP");
     // 保存
     c6->SaveAs("theta_picture/step6_k_function_optimization.png", "png");
     c6->SaveAs("theta_picture/step6_k_function_optimization_high_res.png");
     // 第七步：theta扰动优化
     thetaPerturbationOptimization(data, p_opt[0], p_opt);
 }
 
 // theta扰动优化函数实现
 void thetaPerturbationOptimization(const MeasurementData &data, double k_base, const double* k_params) {
     // k函数模型
     auto k_model = [](double r, double alpha, const double* p) {
         return p[0] * (1 + p[1]*r + p[2]*alpha + p[3]*r*alpha + p[4]*r*r + p[5]*alpha*alpha);
     };
     
     // 1. 定义扰动模型
     auto perturb_model = [](double r, double alpha, const double* p) {
         return p[0] + p[1]*r + p[2]*sin(alpha*TMath::Pi()/180.0) + 
                p[3]*sin(2*alpha*TMath::Pi()/180.0) + p[4]*r*sin(alpha*TMath::Pi()/180.0);
     };
     
     // 2. 定义RMSE目标函数
     auto rmse_func = [&](const double* p) {
         int valid_count = 0;
         double sum2 = 0;
         for (size_t i = 0; i < data.r.size(); ++i) {
             double k_val = k_model(data.r[i], data.alpha[i], k_params);
             std::vector<double> t_calc(1);
             std::vector<bool> v_pt(1);
             MeasurementData single;
             single.r = {data.r[i]};
             single.alpha = {data.alpha[i]};
             single.theta = {data.theta[i]};
             calculateTheta(k_val, single, t_calc, v_pt);
             if (v_pt[0]) {
                 double perturb = perturb_model(data.r[i], data.alpha[i], p);
                 double err = t_calc[0] + perturb - data.theta[i];
                 sum2 += err * err;
                 valid_count++;
             }
         }
         double rmse = (valid_count == 0) ? 1e6 : sqrt(sum2 / valid_count);
         return rmse;
     };
     
     // 3. 多算法多组初值尝试
     double best_rmse = 1e6;
     double best_params[5] = {0};
     std::string best_algo = "";
     std::vector<std::string> algo_list = {"Migrad", "Simplex", "Combined"};
     std::vector<std::vector<double>> init_list = {
         {0, 0, 0, 0, 0},
         {0.5, 0, 0, 0, 0},
         {0, 0.001, 0, 0, 0},
         {0, 0, 0.01, 0, 0},
         {0, 0, 0, 0.01, 0},
         {0, 0, 0, 0, 0.001}
     };
     
     for (const auto& algo : algo_list) {
         for (const auto& init : init_list) {
             ROOT::Math::Minimizer* min = ROOT::Math::Factory::CreateMinimizer("Minuit2", algo.c_str());
             ROOT::Math::Functor f(rmse_func, 5);
             min->SetFunction(f);
             min->SetStrategy(2);
             min->SetVariable(0, "a1", init[0], 1e-2);
             min->SetVariable(1, "a2", init[1], 1e-4);
             min->SetVariable(2, "a3", init[2], 1e-2);
             min->SetVariable(3, "a4", init[3], 1e-2);
             min->SetVariable(4, "a5", init[4], 1e-4);
             min->Minimize();
             const double* p_opt = min->X();
             double rmse = rmse_func(p_opt);
             if (rmse < best_rmse) {
                 best_rmse = rmse;
                 for (int i = 0; i < 5; ++i) best_params[i] = p_opt[i];
                 best_algo = algo;
             }
             delete min;
         }
     }
     
     std::cout << "最优扰动参数: [" << best_params[0] << ", " << best_params[1] << ", " 
               << best_params[2] << ", " << best_params[3] << ", " << best_params[4] 
               << "] -> RMSE: " << best_rmse << " (算法: " << best_algo << ")" << std::endl;
     
     // 将最优扰动参数保存到全局变量，以便主函数使用
     extern double g_perturb_params[5];
     for (int i = 0; i < 5; ++i) {
         g_perturb_params[i] = best_params[i];
     }
     
 }
 
 // theta校正函数优化
 void thetaCorrectionFunctionOptimization(const MeasurementData &data, double k_base, const double* k_params, const double* perturb_params) {
     
     // 设定分界点
     double theta_threshold = 45.0;
     
     // 1. 首先计算原始的theta值（使用k函数和扰动项）
     std::vector<double> theta_calc(data.r.size());
     std::vector<bool> valid_points(data.r.size());
     
     // 扰动模型
     auto perturb_model = [](double r, double alpha, const double* p) {
         return p[0] + p[1]*r + p[2]*sin(alpha*TMath::Pi()/180.0) + 
                p[3]*sin(2*alpha*TMath::Pi()/180.0) + p[4]*r*sin(alpha*TMath::Pi()/180.0);
     };
     
     // 计算每个点的theta值
     for (size_t i = 0; i < data.r.size(); ++i) {
         // 计算k值
         double k_val = k_model(data.r[i], data.alpha[i], k_params);
         
         // 计算基础theta值
         std::vector<double> t_calc(1);
         std::vector<bool> v_pt(1);
         MeasurementData single;
         single.r = {data.r[i]};
         single.alpha = {data.alpha[i]};
         single.theta = {data.theta[i]};
         calculateTheta(k_val, single, t_calc, v_pt);
         
         if (v_pt[0]) {
             // 添加扰动项
             double perturb = perturb_model(data.r[i], data.alpha[i], perturb_params);
             theta_calc[i] = t_calc[0] + perturb;
             valid_points[i] = true;
         } else {
             theta_calc[i] = data.theta[i];
             valid_points[i] = false;
         }
     }
     
     // 2. 分离高低角度数据
     std::vector<int> low_idx, high_idx;
     for (size_t i = 0; i < data.theta.size(); ++i) {
         if (valid_points[i]) {  // 只处理有效点
             if (theta_calc[i] <= theta_threshold) {
                 low_idx.push_back(i);
             } else {
                 high_idx.push_back(i);
             }
         }
     }
     
     // 3. 构建设计矩阵
     // 低角度使用增强的模型（10维特征）
     Eigen::MatrixXd X_low(low_idx.size(), 10);
     Eigen::VectorXd y_low(low_idx.size());
     
     // 特征缩放因子
     double r_scale = 300.0;
     double alpha_scale = 180.0;
     double theta_scale = 90.0;
     
     for (size_t i = 0; i < low_idx.size(); ++i) {
         int idx = low_idx[i];
         double th = theta_calc[idx] / theta_scale;
         double r = data.r[idx] / r_scale;
         double alpha = data.alpha[idx] / alpha_scale;
         
         X_low(i,0) = 1.0;                    // 常数项
         X_low(i,1) = th;                     // theta项
         X_low(i,2) = th*th;                  // theta^2项
         X_low(i,3) = r;                      // r项
         X_low(i,4) = r*r;                    // r^2项
         X_low(i,5) = sin(alpha*TMath::Pi()); // sin(alpha)项
         X_low(i,6) = cos(alpha*TMath::Pi()); // cos(alpha)项
         X_low(i,7) = sin(2*alpha*TMath::Pi()); // sin(2*alpha)项
         X_low(i,8) = r*sin(alpha*TMath::Pi()); // r*sin(alpha)项
         X_low(i,9) = alpha;                  // alpha项
         
         y_low(i) = data.theta[idx] - theta_calc[idx];  // 校正量
     }
     
     // 高角度使用简化模型（7维特征）
     Eigen::MatrixXd X_high(high_idx.size(), 7);
     Eigen::VectorXd y_high(high_idx.size());
     
     for (size_t i = 0; i < high_idx.size(); ++i) {
         int idx = high_idx[i];
         double th = theta_calc[idx] / theta_scale;
         double r = data.r[idx] / r_scale;
         double alpha = data.alpha[idx] / alpha_scale;
         
         X_high(i,0) = 1.0;                    // 常数项
         X_high(i,1) = th;                     // theta项
         X_high(i,2) = th*th;                  // theta^2项
         X_high(i,3) = r;                      // r项
         X_high(i,4) = r*r;                    // r^2项
         X_high(i,5) = sin(alpha*TMath::Pi()); // sin(alpha)项
         X_high(i,6) = sin(2*alpha*TMath::Pi()); // sin(2*alpha)项
         
         y_high(i) = data.theta[idx] - theta_calc[idx];  // 校正量
     }
     
     double lambda_low = 0.001;
     double lambda_high = 0.001;
     
     // 直接对增广矩阵进行SVD求解
     Eigen::VectorXd params_low;
     Eigen::VectorXd params_high;
     
     // 低角度模型求解
     {
         // 构建增广矩阵 [X; sqrt(lambda)*I]
         Eigen::MatrixXd A_low(X_low.rows() + X_low.cols(), X_low.cols());
         Eigen::VectorXd b_low(X_low.rows() + X_low.cols());
         
         A_low.topRows(X_low.rows()) = X_low;
         A_low.bottomRows(X_low.cols()) = sqrt(lambda_low) * Eigen::MatrixXd::Identity(X_low.cols(), X_low.cols());
         
         b_low.head(X_low.rows()) = y_low;
         b_low.tail(X_low.cols()).setZero();
         
         // 使用SVD求解
         Eigen::JacobiSVD<Eigen::MatrixXd> svd_low(A_low, Eigen::ComputeThinU | Eigen::ComputeThinV);
         params_low = svd_low.solve(b_low);
         
         // 检查求解质量
         double relative_error = (A_low * params_low - b_low).norm() / b_low.norm();
         std::cout << "低角度模型相对误差: " << relative_error << std::endl;
         
         // 打印条件数
         double cond = svd_low.singularValues()(0) / 
                      svd_low.singularValues()(svd_low.singularValues().size()-1);
         std::cout << "低角度模型条件数: " << cond << std::endl;
     }
     
     // 高角度模型求解
     {
         // 构建增广矩阵 [X; sqrt(lambda)*I]
         Eigen::MatrixXd A_high(X_high.rows() + X_high.cols(), X_high.cols());
         Eigen::VectorXd b_high(X_high.rows() + X_high.cols());
         
         A_high.topRows(X_high.rows()) = X_high;
         A_high.bottomRows(X_high.cols()) = sqrt(lambda_high) * Eigen::MatrixXd::Identity(X_high.cols(), X_high.cols());
         
         b_high.head(X_high.rows()) = y_high;
         b_high.tail(X_high.cols()).setZero();
         
         // 使用SVD求解
         Eigen::JacobiSVD<Eigen::MatrixXd> svd_high(A_high, Eigen::ComputeThinU | Eigen::ComputeThinV);
         params_high = svd_high.solve(b_high);
         
         // 检查求解质量
         double relative_error = (A_high * params_high - b_high).norm() / b_high.norm();
         std::cout << "高角度模型相对误差: " << relative_error << std::endl;
         
         // 打印条件数
         double cond = svd_high.singularValues()(0) / 
                      svd_high.singularValues()(svd_high.singularValues().size()-1);
         std::cout << "高角度模型条件数: " << cond << std::endl;
     }
     
     // 5. 应用校正函数
     std::vector<double> theta_corrected(data.r.size());
     for (size_t i = 0; i < data.r.size(); ++i) {
         if (!valid_points[i]) {
             theta_corrected[i] = data.theta[i];
             continue;
         }
         
         double th = theta_calc[i] / theta_scale;
         double r = data.r[i] / r_scale;
         double alpha = data.alpha[i] / alpha_scale;
         double correction = 0.0;
         
         if (theta_calc[i] <= theta_threshold) {
             // 低角度校正
             correction = params_low(0) + 
                         params_low(1)*th + 
                         params_low(2)*th*th + 
                         params_low(3)*r + 
                         params_low(4)*r*r + 
                         params_low(5)*sin(alpha*TMath::Pi()) + 
                         params_low(6)*cos(alpha*TMath::Pi()) + 
                         params_low(7)*sin(2*alpha*TMath::Pi()) + 
                         params_low(8)*r*sin(alpha*TMath::Pi()) + 
                         params_low(9)*alpha;
         } else {
             // 高角度校正
             correction = params_high(0) + 
                         params_high(1)*th + 
                         params_high(2)*th*th + 
                         params_high(3)*r + 
                         params_high(4)*r*r + 
                         params_high(5)*sin(alpha*TMath::Pi()) + 
                         params_high(6)*sin(2*alpha*TMath::Pi());
         }
         
         theta_corrected[i] = theta_calc[i] + correction;
     }
     
     // 6. 计算统计信息
     double sum = 0.0, sum2 = 0.0;
     double max_error = -1e10, min_error = 1e10;
     int valid_count = 0;
     
     for (size_t i = 0; i < data.r.size(); ++i) {
         if (valid_points[i]) {
             double error = theta_corrected[i] - data.theta[i];
             sum += error;
             sum2 += error * error;
             max_error = std::max(max_error, error);
             min_error = std::min(min_error, error);
             valid_count++;
         }
     }
     
     double rmse = sqrt(sum2/valid_count);
     double mean = sum/valid_count;
     double stddev = sqrt((sum2/valid_count) - (mean*mean));
     
     // 输出统计信息
     std::cout << "校正后统计信息：" << std::endl;
     std::cout << "RMSE: " << rmse << "°" << std::endl;
     std::cout << "均值: " << mean << "°" << std::endl;
     std::cout << "标准差: " << stddev << "°" << std::endl;
     std::cout << "最大误差: " << max_error << "°" << std::endl;
     std::cout << "最小误差: " << min_error << "°" << std::endl;
     
     // 输出拟合参数
     std::cout << "\n低角度校正参数 (≤" << theta_threshold << "°)：" << std::endl;
     std::cout << "常数项: " << params_low(0) << std::endl;
     std::cout << "theta项: " << params_low(1) << std::endl;
     std::cout << "theta^2项: " << params_low(2) << std::endl;
     std::cout << "r项: " << params_low(3) << std::endl;
     std::cout << "r^2项: " << params_low(4) << std::endl;
     std::cout << "sin(alpha)项: " << params_low(5) << std::endl;
     std::cout << "cos(alpha)项: " << params_low(6) << std::endl;
     std::cout << "sin(2*alpha)项: " << params_low(7) << std::endl;
     std::cout << "r*sin(alpha)项: " << params_low(8) << std::endl;
     std::cout << "alpha项: " << params_low(9) << std::endl;
     
     std::cout << "\n高角度校正参数 (>" << theta_threshold << "°)：" << std::endl;
     std::cout << "常数项: " << params_high(0) << std::endl;
     std::cout << "theta项: " << params_high(1) << std::endl;
     std::cout << "theta^2项: " << params_high(2) << std::endl;
     std::cout << "r项: " << params_high(3) << std::endl;
     std::cout << "r^2项: " << params_high(4) << std::endl;
     std::cout << "sin(alpha)项: " << params_high(5) << std::endl;
     std::cout << "sin(2*alpha)项: " << params_high(6) << std::endl;
     
     // 7. 绘制校正前后对比图
     TCanvas *c8 = new TCanvas("c8", "Theta Correction Function Optimization", 1600, 1200);
     
     // 创建校正前的散点图
     TGraph *g_before = new TGraph();
     int point_idx = 0;
     for (size_t i = 0; i < data.r.size(); ++i) {
         if (valid_points[i]) {
             g_before->SetPoint(point_idx++, data.theta[i], theta_calc[i]);
         }
     }
     g_before->SetMarkerStyle(20);
     g_before->SetMarkerColor(kBlue);
     g_before->SetMarkerSize(1.0);
     g_before->SetTitle("Theta Correction: Measured vs Calculated");
     g_before->GetXaxis()->SetTitle("Measured Theta (deg)");
     g_before->GetYaxis()->SetTitle("Calculated Theta (deg)");
     
     // 创建校正后的散点图
     TGraph *g_after = new TGraph();
     point_idx = 0;
     for (size_t i = 0; i < data.r.size(); ++i) {
         if (valid_points[i]) {
             g_after->SetPoint(point_idx++, data.theta[i], theta_corrected[i]);
         }
     }
     g_after->SetMarkerStyle(21);
     g_after->SetMarkerColor(kRed);
     g_after->SetMarkerSize(1.0);
     
     // 绘制图形
     g_before->Draw("AP");
     g_after->Draw("P SAME");
     
     // 添加理想线 y=x
     TLine *ideal_line = new TLine(0, 0, 90, 90);
     ideal_line->SetLineStyle(2);
     ideal_line->SetLineColor(kBlack);
     ideal_line->Draw();
     
     // 添加图例
     TLegend *legend = new TLegend(0.15, 0.7, 0.45, 0.85);
     legend->AddEntry(g_before, "Before Correction", "p");
     legend->AddEntry(g_after, "After Correction", "p");
     legend->AddEntry(ideal_line, "Ideal Line", "l");
     legend->Draw();
     
     // 设置坐标轴范围
     g_before->GetXaxis()->SetRangeUser(0, 90);
     g_before->GetYaxis()->SetRangeUser(0, 90);
     
     // 保存图像
     c8->SaveAs("theta_picture/step8_correction_function.png", "png");
     c8->SaveAs("theta_picture/step8_correction_function_high_res.png");
 }
 
 // 主函数
 int ROOT_oblique_projection_motion_fitting() {
     // 设置ROOT样式
     gStyle->SetOptStat(0);
     gStyle->SetPalette(kRainBow);
     
     // 提升图片质量
     gStyle->SetLineWidth(2);
     gStyle->SetFrameLineWidth(2);
     gStyle->SetHistLineWidth(2);
     gStyle->SetCanvasColor(0);
     gStyle->SetCanvasBorderMode(0);
     gStyle->SetPadColor(0);
     gStyle->SetPadBorderMode(0);
     gStyle->SetFrameFillColor(0);
     
     // 启用OpenGL渲染
     gStyle->SetCanvasPreferGL(true);
     
     // 设置抗锯齿
     gStyle->SetHistFillStyle(1001);
     gStyle->SetHistLineStyle(1);
     gStyle->SetMarkerStyle(20);
     
     // 设置高DPI支持
     gStyle->SetImageScaling(2.0);  // 使用2倍DPI，这是一个比较合理的值
     
     // 设置字体和标记
     gStyle->SetTextSize(0.035);
     gStyle->SetLabelSize(0.035, "xyz");
     gStyle->SetTitleSize(0.04, "xyz");
     gStyle->SetTitleOffset(1.1, "xyz");
     gStyle->SetMarkerSize(1.0);
     
     // 设置中文字体支持
     gROOT->ProcessLine(".x $ROOTSYS/tutorials/graphs/rootlogon.C");
     gStyle->SetTitleFont(132, "xyz");
     gStyle->SetLabelFont(132, "xyz");
     gStyle->SetTextFont(132);
     gStyle->SetLegendFont(132);
     
     // 设置直方图样式
     gStyle->SetBarWidth(0.8);
     gStyle->SetBarOffset(0.1);
     
     // 创建theta_picture目录
     system("mkdir -p theta_picture");
     
     // 加载实验数据
     MeasurementData data;
     loadData(data);
     
     std::cout << "=== Theta Angle Correction System C++/ROOT Implementation ===" << std::endl;
     
     // 第一步：多项式拟合
     std::cout << "\n=== Step 1: Polynomial Fitting ===" << std::endl;
     polynomialFitting(data);
     
     // 第二步：大范围搜索k
     std::cout << "\n=== Step 2: Large Range k Search ===" << std::endl;
     double k_min = 0.0001;
     double k_max = 0.01;
     double k_opt = kSearch(data, k_min, k_max);
     
     // 第三步：解析式拟合曲面
     std::cout << "\n=== Step 3: Analytical Surface Fitting ===" << std::endl;
     analyticalSurface(data, k_opt);
     
     // 第四步：精细k值搜索
     std::cout << "\n=== Step 4: Fine Tuning k Value ===" << std::endl;
     double k_opt_final = k_opt;
     fineTuneK(data, k_opt);
     
     // 第五步：残差分析
     std::cout << "\n=== Step 5: Residual Analysis ===" << std::endl;
     residualAnalysis(data, k_opt_final);
     
     // 第六步：k函数优化
     std::cout << "\n=== Step 6: k Function Optimization ===" << std::endl;
     double k_params[6] = {k_opt_final, 0, 0, 0, 0, 0}; // 初始化k函数参数
     kFunctionOptimization(data, k_opt_final);
     
     // 第七步：theta扰动优化
     std::cout << "\n=== Step 7: Theta Perturbation Optimization ===" << std::endl;
     thetaPerturbationOptimization(data, k_opt_final, k_params);
     
     // 第八步：theta校正函数优化
     std::cout << "\n=== Step 8: Theta Correction Function Optimization ===" << std::endl;
     thetaCorrectionFunctionOptimization(data, k_opt_final, k_params, g_perturb_params); // 使用全局变量
     
     std::cout << "\nProgram Completed!" << std::endl;
     std::cout << "Figures have been saved to theta_picture directory" << std::endl;
     
     return 0;
 }
 