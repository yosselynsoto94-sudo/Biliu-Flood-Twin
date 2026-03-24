import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
import joblib
import os
import warnings

# ================= 新增：强大的 GIS 与空间插值引擎 =================
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pykrige.uk import UniversalKriging
from shapely.geometry import Point

warnings.filterwarnings('ignore')
# 设置中文字体，防止图表中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 核心路径配置区 (已完全固定) =================
XAJ_PARAMS_FILE = r"Smart_Flood_App/data/best_xaj_15params_DE.csv"
LSTM_MODEL_PATH = r"Smart_Flood_App/data/lstm_model_DE.pth"
SCALER_FILE = r"Smart_Flood_App/data/scalers_DE.pkl"

# 空间地理数据
STATION_INFO_FILE = r"Smart_Flood_App/data/流域雨量站位置.csv"
DEM_FILE = r"Smart_Flood_App/data/12.5m.tif"
SHP_YUSHI = r"Smart_Flood_App/data/玉石水库范围.shp"
SHP_INTERVAL = r"Smart_Flood_App/data/区间范围.shp"


# ================= 2. 物理模型与深度学习架构 =================
def run_xaj_1h(P, E, params, area):
    """15参数新安江模型 (含12小时物理汇流约束)"""
    K, IMP, B, WM, WUM, WLM, C, SM, EX, KI, KG, CI, CG, CS, L = params
    rem_S = (1.0 - CS) ** (1.0 / 12.0)
    rem_I = CI ** (1.0 / 24.0)
    rem_G = CG ** (1.0 / 24.0)
    KI_h = KI / 24.0
    KG_h = KG / 24.0
    WU, WL, WD = WUM * 0.8, WLM * 0.8, (WM - WUM - WLM) * 0.8
    W = WU + WL + WD
    S = SM * 0.5
    Q_sim = np.zeros(len(P))
    qs, qi, qg = 0.0, 0.0, 0.05

    for t in range(len(P)):
        P_i = P[t]
        E_p = (E[t] * K) / 24.0
        if P_i + WU >= E_p:
            EU, EL, ED = E_p, 0.0, 0.0
            WU = WU + P_i - E_p
        else:
            EU = P_i + WU
            WU = 0.0
            if WL >= C * WLM:
                EL = (E_p - EU) * (WL / WLM)
                ED = 0.0
            elif WL >= C * (E_p - EU):
                EL = C * (E_p - EU)
                ED = 0.0
            else:
                EL = WL
                ED = C * (E_p - EU) - EL
        WL = max(0.0, WL - EL)
        WD = max(0.0, WD - ED)
        PE = P_i - (EU + EL + ED)
        R = 0.0
        if PE > 0:
            a = WM * (1.0 - (1.0 - W / WM) ** (1.0 / (1.0 + B))) if W < WM else WM
            if a + PE < WM:
                R = PE - (WM - W) + WM * (1.0 - (a + PE) / WM) ** (1.0 + B)
            else:
                R = PE - (WM - W)
            R = R * (1 - IMP) + PE * IMP
            W = W + PE - R
            WU = WU + PE - R
            if WU > WUM:
                WL = WL + WU - WUM
                WU = WUM
                if WL > WLM:
                    WD = WD + WL - WLM
                    WL = WLM
                    if WD > (WM - WUM - WLM):
                        WD = (WM - WUM - WLM)
            W = WU + WL + WD
        RS = 0.0
        if R > 0:
            FR = R / PE if PE > 0 else 0.01
            FR = min(max(FR, 0.01), 1.0)
            MS = SM * (1.0 + EX)
            AU = MS * (1.0 - (1.0 - S / SM) ** (1.0 / (1.0 + EX))) if S < SM else MS
            R_FR = R / FR
            if AU + R_FR < MS:
                RS_FR = R_FR - (SM - S) + SM * (1.0 - (AU + R_FR) / MS) ** (1.0 + EX)
            else:
                RS_FR = R_FR - (SM - S)
            RS = RS_FR * FR
            S = S + R - RS
        RI = S * KI_h
        RG = S * KG_h
        S = max(0.0, S - RI - RG)
        qs = qs * rem_S + RS * (1.0 - rem_S)
        qi = qi * rem_I + RI * (1.0 - rem_I)
        qg = qg * rem_G + RG * (1.0 - rem_G)
        Q_sim[t] = (qs + qi + qg) * area / 3.6
    return Q_sim


class ResidualLSTM(nn.Module):
    """LSTM隐式水库调度代偿网络"""

    def __init__(self, input_size=7, hidden_size=32, num_layers=1, output_size=1):
        super(ResidualLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# ================= 3. 高级空间特征提取引擎 (协同-克里金自动高程版) =================
class KrigingSpatialEngine:
    def __init__(self, dem_path, shp_yushi_path, shp_interval_path):
        self.dem_path = dem_path
        self.shp_yushi = gpd.read_file(shp_yushi_path)
        self.shp_interval = gpd.read_file(shp_interval_path)

        self.yu_x, self.yu_y, self.yu_z = self._extract_grid_from_dem(self.shp_yushi)
        self.int_x, self.int_y, self.int_z = self._extract_grid_from_dem(self.shp_interval)

    def _extract_grid_from_dem(self, shp):
        with rasterio.open(self.dem_path) as src:
            out_image, out_transform = mask(src, shp.geometry, crop=True)
            out_image = out_image[0]

            valid_mask = out_image != src.nodata
            rows, cols = np.where(valid_mask)
            xs, ys = rasterio.transform.xy(out_transform, rows, cols)
            zs = out_image[valid_mask]

            return np.array(xs), np.array(ys), zs

    def _get_elevation_from_dem(self, lon_arr, lat_arr):
        coords = [(lon, lat) for lon, lat in zip(lon_arr, lat_arr)]
        elevations = []
        with rasterio.open(self.dem_path) as src:
            for val in src.sample(coords):
                elevations.append(val[0])
        return np.array(elevations)

    def calculate_hourly_spatial_features(self, df_storm, df_stations, progress_bar=None):
        storm_len = len(df_storm)
        P_Y, P_I, Cv_Y, Cv_I = np.zeros(storm_len), np.zeros(storm_len), np.zeros(storm_len), np.zeros(storm_len)

        available_stations = [col for col in df_storm.columns if col in df_stations['Station'].values]
        st_lon = df_stations.set_index('Station').loc[available_stations, 'Lon'].values
        st_lat = df_stations.set_index('Station').loc[available_stations, 'Lat'].values
        st_elv = self._get_elevation_from_dem(st_lon, st_lat)

        for t in range(storm_len):
            precip = df_storm[available_stations].iloc[t].values

            if np.all(precip < 0.1):
                P_Y[t], P_I[t], Cv_Y[t], Cv_I[t] = 0.0, 0.0, 0.0, 0.0
            else:
                try:
                    uk = UniversalKriging(
                        st_lon, st_lat, precip,
                        variogram_model='linear',
                        drift_terms=['specified'],
                        specified_drift=[st_elv]
                    )

                    z_yu, _ = uk.execute('points', self.yu_x, self.yu_y, specified_drift_arrays=[self.yu_z])
                    P_Y[t] = np.mean(z_yu)
                    Cv_Y[t] = np.std(z_yu) / (P_Y[t] + 1e-6)

                    z_int, _ = uk.execute('points', self.int_x, self.int_y, specified_drift_arrays=[self.int_z])
                    P_I[t] = np.mean(z_int)
                    Cv_I[t] = np.std(z_int) / (P_I[t] + 1e-6)

                except Exception as e:
                    P_Y[t], P_I[t], Cv_Y[t], Cv_I[t] = np.mean(precip), np.mean(precip), 0.1, 0.1

            if progress_bar:
                progress_bar.progress((t + 1) / storm_len)

        return P_Y, P_I, Cv_Y, Cv_I


# ================= 4. 模型加载与物理耦合推演引擎 =================
@st.cache_resource
def load_all_assets():
    param_df = pd.read_csv(XAJ_PARAMS_FILE, encoding='gbk')
    params_y = param_df['Yushi_Value'].values
    params_i = param_df['Interval_Value'].values

    model = ResidualLSTM()
    model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location='cpu'))
    model.eval()

    scaler_x, scaler_y = joblib.load(SCALER_FILE)
    df_stations = pd.read_csv(STATION_INFO_FILE, encoding='gbk')

    spatial_engine = KrigingSpatialEngine(DEM_FILE, SHP_YUSHI, SHP_INTERVAL)

    return params_y, params_i, model, scaler_x, scaler_y, df_stations, spatial_engine


def execute_flood_routing(P_Y, P_I, Cv_Y, Cv_I, params_y, params_i, model, scaler_x, scaler_y, lag_yushi,
                          recession_hours):
    """
    🌟 核心更新：引入动态退水时长机制
    不再硬性限制为 144 小时，而是根据真实的降雨长度，自动叠加退水延拓时长。
    """
    storm_len = len(P_Y)
    TOTAL_HOURS = storm_len + recession_hours  # 动态计算总步长

    # 时序延拓至退水期完毕
    P_Y_full = np.concatenate([P_Y, np.zeros(recession_hours)])
    P_I_full = np.concatenate([P_I, np.zeros(recession_hours)])
    Cv_Y_full = np.concatenate([Cv_Y, np.zeros(recession_hours)])
    Cv_I_full = np.concatenate([Cv_I, np.zeros(recession_hours)])

    E_Y_full = np.full(TOTAL_HOURS, 0.1)
    E_I_full = np.full(TOTAL_HOURS, 0.1)

    q_xaj_y_raw = run_xaj_1h(P_Y_full, E_Y_full, params_y, 313)
    q_xaj_i = run_xaj_1h(P_I_full, E_I_full, params_i, 1772)
    q_xaj_y = np.concatenate([np.full(lag_yushi, q_xaj_y_raw[0]), q_xaj_y_raw])[:TOTAL_HOURS]
    q_xaj_total = q_xaj_y + q_xaj_i

    P_Y_roll = pd.Series(P_Y_full).rolling(window=24, min_periods=1).sum().values
    P_I_roll = pd.Series(P_I_full).rolling(window=24, min_periods=1).sum().values
    E_Y_roll = pd.Series(E_Y_full).rolling(window=24, min_periods=1).sum().values
    E_I_roll = pd.Series(E_I_full).rolling(window=24, min_periods=1).sum().values
    Q_sim_roll = pd.Series(q_xaj_total).rolling(window=24, min_periods=1).mean().values

    features_daily_equiv = np.column_stack([
        P_Y_roll, P_I_roll, E_Y_roll, E_I_roll, Cv_Y_full, Cv_I_full, Q_sim_roll
    ])

    features_scaled = scaler_x.transform(features_daily_equiv)
    features_scaled = np.clip(features_scaled, -3.0, 3.0)

    q_residuals_scaled = []
    for t in range(TOTAL_HOURS):
        seq = []
        for step in [4, 3, 2, 1, 0]:
            idx = max(0, t - step * 24)
            seq.append(features_scaled[idx])
        seq = np.array(seq)
        x_in = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            res_scaled = model(x_in).numpy().flatten()[0]
            q_residuals_scaled.append(res_scaled)

    q_residuals = scaler_y.inverse_transform(np.array(q_residuals_scaled).reshape(-1, 1)).flatten()

    max_allowable_up = q_xaj_total * 0.05
    max_allowable_down = q_xaj_total * 0.10
    q_residuals_clipped = np.clip(q_residuals, -max_allowable_down, max_allowable_up)

    q_final = q_xaj_total + q_residuals_clipped

    if len(q_final) > storm_len:
        q_smooth = savgol_filter(q_final, 11, 2)
        safe_idx = storm_len + 8
        if safe_idx < len(q_final):
            q_final[safe_idx:] = q_smooth[safe_idx:]

    return np.maximum(q_final, 0), TOTAL_HOURS


# ================= 5. 前端网页 UI 架构 =================
st.set_page_config(page_title="碧流河智能防洪数字孪生平台", layout="wide", page_icon="🌊")

st.markdown("""
<div style="background-color:#0b1c2c; padding:25px; border-radius:12px; margin-bottom:25px; border-left: 6px solid #1a80e5;">
    <h1 style="color:#ffffff; margin-top:0;">🌊 碧流河水库防洪决策数字孪生平台</h1>
    <p style="font-size:16px; color:#a2b1c6;">
    <b>端到端集成架构：</b>系统底层集成 DEM 高程矩阵与协同克里金插值引擎（Co-Kriging）。用户上传任意时长的离散站点降水数据后，平台将自动解析空间异质性特征，并驱动 XAJ-LSTM 双引擎耦合网络动态顺延推演，输出高精度水库入库洪水过程与 3D 态势感知。
    </p>
</div>
""", unsafe_allow_html=True)

# 侧边栏：模板下载、文件上传与参数设置
with st.sidebar:
    st.header("⚙️ 气象站点数据接入")

    st.markdown("### 📄 第 1 步：准备输入数据")
    template_data = {
        'Time(Hour)': [1, 2, 3, 4, 5],
        '碧流河水库站': [0.0, 12.5, 34.2, 15.0, 2.1],
        '大姜屯站': [0.0, 8.5, 22.1, 14.5, 3.0],
        '桂云花站': [0.0, 15.0, 41.5, 26.2, 11.5],
        '茧场站': [0.0, 11.0, 18.5, 33.0, 9.5],
        '矿洞沟站': [0.0, 5.5, 12.1, 8.5, 1.0],
        '孟家店站': [0.0, 10.2, 25.4, 18.0, 4.2],
        '太平庄站': [0.0, 14.1, 38.2, 22.5, 8.6],
        '天益站': [0.0, 9.5, 20.1, 16.2, 5.1],
        '西扒山站': [0.0, 16.5, 45.0, 28.5, 12.0],
        '小石棚站': [0.0, 7.8, 19.5, 12.4, 2.5]
    }
    df_template = pd.DataFrame(template_data)

    with st.expander("👀 查看所需数据格式示例及说明"):
        st.markdown("""
        **1. 表格格式**：第一列必须为时间，其余每一列为真实的雨量站名称，填入逐时降水(mm)。

        **2. ⏳ 时长完全无限制**：
        您可以上传**任意小时数**的降水表格（例如13小时、55小时或200小时）。引擎会自动计算您的暴雨时长，并结合您在下方设定的“退水期”，动态生成完整的洪水过程线。
        """)
        st.dataframe(df_template)

    csv_template = df_template.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 下载空白输入模板 (CSV)", data=csv_template, file_name='雨量站逐时降水输入模板.csv',
                       mime='text/csv')

    st.markdown("---")
    st.markdown("### 📤 第 2 步：控制与推演")
    uploaded_file = st.file_uploader("📂 上传您准备好的降雨文件", type="csv")

    # 🌟 新增：用户自定义退水期长度
    st.markdown("#### 🎛️ 物理干预参数")
    recession_time = st.slider("💧 暴雨后退水期延拓时长 (h)", min_value=24, max_value=168, value=72, step=12,
                               help="在降雨结束后，系统继续向后推演的小时数，以观测洪水完全消退的过程。")
    lag_time = st.slider("⏱️ 干流洪水演进滞后时间 (h)", 1, 6, 3)

try:
    with st.spinner("⏳ 正在挂载底层 GIS 引擎与深度学习矩阵..."):
        params_y, params_i, lstm_model, scaler_x, scaler_y, df_stations, spatial_engine = load_all_assets()
        system_ready = True
except Exception as e:
    st.error(f"❌ 底层服务启动失败，请检查文件路径或联系管理员: {e}")
    system_ready = False

if uploaded_file is not None and system_ready:
    try:
        df_storm = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df_storm = pd.read_csv(uploaded_file, encoding='gbk')

    storm_len = len(df_storm)
    st.success(f"✅ 成功接入气象序列，时段长度: {storm_len} 小时。")

    if st.button("🚀 启动数字孪生时空演进引擎", type="primary", use_container_width=True):
        st.markdown("### 🗺️ 阶段一：多维空间降水场重建")
        progress_bar = st.progress(0)
        with st.spinner("正在逐时计算 DEM 网格点高程漂移并提取异质性空间特征..."):
            P_Y, P_I, Cv_Y, Cv_I = spatial_engine.calculate_hourly_spatial_features(df_storm, df_stations, progress_bar)
            st.write("✔️ 空间变差系数 (Cv) 及流域面雨量动态重建完成。")

        st.markdown("### 🧠 阶段二：物理-数据双驱动水文演进")
        with st.spinner("调度 15 参数物理模型及 LSTM 水库隐式调洪网络..."):
            # 获取动态计算的总时间总长 total_hours
            q_final, total_hours = execute_flood_routing(
                P_Y, P_I, Cv_Y, Cv_I, params_y, params_i, lstm_model, scaler_x, scaler_y, lag_time, recession_time
            )
            st.write(f"✔️ 全周期非线性洪峰推演完成（总计算历时：{total_hours} 小时）。")

        # --- 第一部分：二维过程线展示 ---
        st.markdown("---")
        st.markdown("### 📊 碧流河水库总入库洪水成果过程线")
        fig = plt.figure(figsize=(14, 6), dpi=120)
        plt.fill_between(range(1, total_hours + 1), q_final, color='#3498db', alpha=0.2)
        plt.plot(range(1, total_hours + 1), q_final, label='水库智能预报入库洪水', color='#e74c3c', linewidth=2.5)
        plt.axvline(x=storm_len, color='gray', linestyle='--', alpha=0.8, label=f'降水事件终止 ({storm_len}h)')
        plt.xlabel('时间序列 (小时)', fontsize=12)
        plt.ylabel('演进流量 (m³/s)', fontsize=12)
        plt.xlim(1, total_hours)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=12, loc='upper right')
        st.pyplot(fig)

        st.balloons()

        # 结果打包提供下载 (长度也变成动态的 total_hours)
        df_result = pd.DataFrame({'Hour': range(1, total_hours + 1), 'Inflow_Prediction(m3/s)': np.round(q_final, 2)})
        csv_res = df_result.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 保存高精度洪水预报数据 (CSV)", data=csv_res, file_name='Smart_Flood_Result.csv',
                           mime='text/csv')
else:
    if system_ready:
        st.info("👆 请在系统侧边栏下载模板、填入数据并上传，以激活计算引擎。")
