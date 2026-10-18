import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from requests import Session
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
from fredapi import Fred

# 網頁版面設定
st.set_page_config(page_title="三合一多元資產量化配置與動態對沖儀表板", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        background-color: #0d1b2a;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .section-header {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
        background-color: #1b263b;
        padding: 8px 12px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🌐 三合一多元資產量化配置系統 (美股選股 + 跨資產ETF + 公債總經預測)</div>', unsafe_allow_html=True)

# ==========================================
# 0. 帳號密碼登入驗證保護機制
# ==========================================
def check_password():
    if st.session_state.get("password_correct", False):
        return True

    st.markdown("## 🔐 系統登入驗證")
    st.info("請輸入您的專屬帳號與密碼以存取多元資產配置系統。")
    
    with st.form("login_form"):
        username = st.text_input("帳號 (Username)")
        password = st.text_input("密碼 (Password)", type="password")
        submit_btn = st.form_submit_button("登入")

    if submit_btn:
        passwords_dict = st.secrets.get("passwords", {})
        if username in passwords_dict and passwords_dict[username] == password:
            st.session_state["password_correct"] = True
            st.session_state["logged_in_user"] = username
            st.rerun()
        else:
            st.error("❌ 帳號或密碼錯誤，請重新輸入。")
            
    return False

if not check_password():
    st.stop()

st.success(f"歡迎回來，{st.session_state.get('logged_in_user')}！")

if st.sidebar.button("🔒 登出系統"):
    st.session_state["password_correct"] = False
    st.rerun()

# ==========================================
# 1. 字典與對照表定義
# ==========================================
US_STOCK_MAP = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "META": "Meta Platforms", "BRK-B": "Berkshire", "LLY": "Eli Lilly",
    "AVGO": "Broadcom", "TSLA": "Tesla", "WMT": "Walmart", "JPM": "JPMorgan",
    "V": "Visa", "XOM": "ExxonMobil", "MA": "Mastercard", "COST": "Costco",
    "NFLX": "Netflix", "HD": "Home Depot", "PG": "Procter & Gamble", "JNJ": "Johnson & Johnson"
}
US_KEYS = list(US_STOCK_MAP.keys())

ETF_STOCK_MAP = {
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "TLT": "20+年期公債 ETF",
    "GLD": "黃金信託 ETF", "IWM": "羅素2000 ETF", "VNQ": "美國房地產 ETF",
    "SMH": "半導體 ETF", "HYG": "高收益債 ETF", "IEF": "7-10年公債 ETF"
}
ETF_KEYS = list(ETF_STOCK_MAP.keys())

# ==========================================
# 側邊欄參數設定
# ==========================================
st.sidebar.header("⚙️ 全域回測參數設定")

default_fred_key = ""
try:
    if "FRED_API_KEY" in st.secrets:
        default_fred_key = st.secrets["FRED_API_KEY"]
except Exception:
    pass

fred_api_key = st.sidebar.text_input("FRED API Key (公債模組必填)", type="password", value=default_fred_key)
if not fred_api_key:
    st.sidebar.warning("⚠️ 請輸入 FRED API Key 以執行公債總經預測模組。")

train_window = st.sidebar.slider("機器學習訓練期數 (Train Window)", min_value=12, max_value=60, value=36, step=6)
target_start_date = st.sidebar.date_input("回測開始日期", pd.to_datetime("2024-01-31"))
target_end_date = st.sidebar.date_input("回測結束日期", pd.to_datetime("2026-12-31"))

st.sidebar.markdown("---")
st.sidebar.markdown("##### ⚖️ 三大策略配置權重 (固定各 33.3%)")
st.sidebar.info("1. 美國多因子機器學習選股 (1/3)\n2. 跨資產 ETF 輪動配置 (1/3)\n3. 債券總經動態預測策略 (1/3)")

run_btn = st.sidebar.button("🚀 開始執行三合一組合回測")

if "multi_exec" not in st.session_state:
    st.session_state.multi_exec = False

if run_btn:
    if not fred_api_key:
        st.error("❌ 請先在側邊欄輸入 FRED API Key！")
    else:
        with st.spinner("正在同步三大模組數據、執行機器學習運算並合成綜合投資組合中..."):
            session = Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            
            fetch_start = pd.to_datetime("2020-01-01")
            fetch_end = pd.to_datetime("2028-12-31")

            # ==========================================
            # 模組一：美股多因子選股策略
            # ==========================================
            us_stocks = US_KEYS[:20]
            stock_fundamentals = {}
            for t in us_stocks:
                stock_fundamentals[t] = {"Value": 0.05, "Quality": 0.15, "Size": 25.0}

            all_us_tickers = us_stocks + ["^GSPC"]
            df_us_raw = yf.download(all_us_tickers, start=fetch_start.strftime("%Y-%m-%d"), end=fetch_end.strftime("%Y-%m-%d"), progress=False, session=session)
            df_us_prices = df_us_raw["Adj Close"] if "Adj Close" in df_us_raw.columns else df_us_raw["Close"]
            
            bm_us_ret = df_us_prices["^GSPC"].resample("ME").last().pct_change().shift(-1)
            us_stock_prices = df_us_prices[[c for c in us_stocks if c in df_us_prices.columns]].ffill().bfill()
            
            df_us_monthly = us_stock_prices.resample("ME").last()
            df_us_next_ret = df_us_monthly.pct_change().shift(-1)
            df_us_mom = df_us_monthly.pct_change(12)
            df_us_vol = (us_stock_prices.pct_change().rolling(252).std() * np.sqrt(252)).resample("ME").last()

            us_dataset = []
            us_valid_dates = df_us_monthly.index[12:-1]
            for date in us_valid_dates:
                for ticker in us_stocks:
                    if ticker not in df_us_monthly.columns: continue
                    mom_v = df_us_mom.loc[date, ticker] if date in df_us_mom.index else np.nan
                    vol_v = df_us_vol.loc[date, ticker] if date in df_us_vol.index else np.nan
                    ret_v = df_us_next_ret.loc[date, ticker] if date in df_us_next_ret.index else np.nan
                    if pd.isna(mom_v) or pd.isna(vol_v) or pd.isna(ret_v): continue
                    fund = stock_fundamentals[ticker]
                    us_dataset.append({
                        "Date": date, "Stock": ticker,
                        "Value": fund["Value"] * 100, "Momentum": mom_v * 100,
                        "Quality": fund["Quality"] * 100, "LowVol": vol_v * 100, "Size": fund["Size"],
                        "Next_Return": ret_v * 100
                    })
            
            df_us_ml = pd.DataFrame(us_dataset).set_index(["Date", "Stock"])
            base_feats = ["Value", "Momentum", "Quality", "LowVol", "Size"]
            df_us_ml[base_feats] = df_us_ml.groupby(level="Date")[base_feats].transform(lambda g: g.rank(pct=True, method='average') * 2 - 1).fillna(0)

            poly = PolynomialFeatures(degree=2, include_bias=False)
            us_exp_list = []
            for dt, grp in df_us_ml.groupby(level="Date"):
                feat_arr = poly.fit_transform(grp[base_feats])
                fn = poly.get_feature_names_out(base_feats)
                g_exp = pd.DataFrame(feat_arr, index=grp.index, columns=fn)
                g_exp["Next_Return"] = grp["Next_Return"]
                us_exp_list.append(g_exp)
            df_us_expanded = pd.concat(us_exp_list)
            fn_us = poly.get_feature_names_out(base_feats)

            us_dates = df_us_expanded.index.get_level_values("Date").unique().sort_values()
            us_port_ret = []
            n_top = max(1, int(len(us_stocks) * 0.2))
            alphas = np.logspace(-2, 4, 10)

            for t in range(train_window, len(us_dates)):
                test_dt = us_dates[t]
                tr_dt = us_dates[t - train_window : t]
                tr_d = df_us_expanded.loc[tr_dt]
                te_d = df_us_expanded.loc[test_dt]
                
                model = RidgeCV(alphas=alphas).fit(tr_d[fn_us], tr_d["Next_Return"])
                preds = model.predict(te_d[fn_us])
                res = pd.DataFrame({"Stock": te_d.index.get_level_values("Stock"), "Actual": te_d["Next_Return"].values, "Pred": preds})
                res["Rank"] = res["Pred"].rank(ascending=False, method='first')
                long_df = res[res["Rank"] <= n_top]
                us_port_ret.append({"Date": test_dt, "Strategy_Return": long_df["Actual"].mean()})

            df_strat_us = pd.DataFrame(us_port_ret).set_index("Date")

            # ==========================================
            # 模組二：跨資產 ETF 配置策略
            # ==========================================
            etfs = ETF_KEYS[:9]
            df_etf_raw = yf.download(etfs, start=fetch_start.strftime("%Y-%m-%d"), end=fetch_end.strftime("%Y-%m-%d"), progress=False, session=session)
            df_etf_prices = df_etf_raw["Adj Close"] if "Adj Close" in df_etf_raw.columns else df_etf_raw["Close"]
            df_etf_monthly = df_etf_prices.resample("ME").last().ffill().bfill()
            df_etf_ret = df_etf_monthly.pct_change().shift(-1) * 100
            df_etf_mom = df_etf_monthly.pct_change(3) # 3個月動能輪動

            etf_port_ret = []
            etf_dates = df_etf_monthly.index[3:-1]
            for dt in etf_dates:
                mom_slice = df_etf_mom.loc[dt].dropna()
                top_etfs = mom_slice.nlargest(3).index.tolist() # 選出動能最強的前 3 檔 ETF
                ret_slice = df_etf_ret.loc[dt, top_etfs]
                etf_port_ret.append({"Date": dt, "Strategy_Return": ret_slice.mean()})
            
            df_strat_etf = pd.DataFrame(etf_port_ret).set_index("Date")

            # ==========================================
            # 模組三：債券總經預測策略 (TLT / TBT)
            # ==========================================
            fred = Fred(api_key=fred_api_key.strip())
            unrate = fred.get_series('UNRATE')
            t5yie = fred.get_series('T5YIE')
            wei = fred.get_series('WEI')

            macro_raw = pd.DataFrame({'Unemployment_Rate': unrate, 'Inflation_Expectation': t5yie, 'WEI': wei}).resample("ME").last().ffill()
            
            bond_tickers = ["^TYX", "TLT", "TBT", "^MOVE"]
            df_bond_raw = yf.download(bond_tickers, start=fetch_start.strftime("%Y-%m-%d"), end=fetch_end.strftime("%Y-%m-%d"), progress=False, session=session)
            df_bond_prices = df_bond_raw["Adj Close"] if "Adj Close" in df_bond_raw.columns else df_bond_raw["Close"]
            
            bond_m = df_bond_prices.resample("ME").last().ffill()
            macro_full = pd.concat([macro_raw, bond_m], axis=1).dropna()
            
            macro_full["Target_Delta"] = macro_full["^TYX"].shift(-3) - macro_full["^TYX"] # 預測未來3個月殖利率變化
            bond_dates = macro_full.index[train_window:-3]
            
            bond_port_ret = []
            features = ["Unemployment_Rate", "Inflation_Expectation", "WEI", "^MOVE"]
            tlt_fwd_ret = df_bond_prices["TLT"].resample("ME").last().pct_change().shift(-1) * 100
            tbt_fwd_ret = df_bond_prices["TBT"].resample("ME").last().pct_change().shift(-1) * 100

            for dt in bond_dates:
                tr_sub = macro_full.loc[:dt].iloc[-train_window-3:-3].dropna(subset=features + ["Target_Delta"])
                te_row = macro_full.loc[dt, features].values.reshape(1, -1)
                
                model_b = RidgeCV(alphas=alphas).fit(tr_sub[features], tr_sub["Target_Delta"])
                pred_delta = model_b.predict(te_row)[0]
                
                # 預期殖利率下降 -> 買入 TLT (多頭)；預期殖利率上升 -> 買入 TBT (空頭)
                strat_ret = tlt_fwd_ret.loc[dt] if pred_delta < 0 else tbt_fwd_ret.loc[dt]
                bond_port_ret.append({"Date": dt, "Strategy_Return": strat_ret})

            df_strat_bond = pd.DataFrame(bond_port_ret).set_index("Date")

            # ==========================================
            # 綜合投資組合合成 (Equal Weight 1/3 配置)
            # ==========================================
            combined = pd.DataFrame({
                "US_Stock": df_strat_us["Strategy_Return"],
                "ETF_Rot": df_strat_etf["Strategy_Return"],
                "Bond_Pred": df_strat_bond["Strategy_Return"]
            }).dropna()

            combined = combined.loc[pd.to_datetime(target_start_date):pd.to_datetime(target_end_date)]
            combined["Combined_Return"] = combined[["US_Stock", "ETF_Rot", "Bond_Pred"]].mean(axis=1)

            combined["US_Nav"] = (1 + combined["US_Stock"] / 100).cumprod()
            combined["ETF_Nav"] = (1 + combined["ETF_Rot"] / 100).cumprod()
            combined["Bond_Nav"] = (1 + combined["Bond_Pred"] / 100).cumprod()
            combined["Combined_Nav"] = (1 + combined["Combined_Return"] / 100).cumprod()

            st.session_state.combined_results = combined
            st.session_state.multi_exec = True

if st.session_state.get("multi_exec", False):
    res_df = st.session_state.combined_results

    st.markdown('<div class="section-header">📊 三合一多元資產配置績效與風險比較摘要</div>', unsafe_allow_html=True)

    def get_metrics(nav_series, ret_series):
        total_days = (res_df.index[-1] - res_df.index[0]).days
        years = max(total_days / 365.25, 0.5)
        cagr = (nav_series.iloc[-1] ** (1 / years)) - 1.0
        mdd = ((nav_series - nav_series.cummax()) / nav_series.cummax()).min()
        return cagr * 100, mdd * 100, ret_series.mean() * 12

    cagr_com, mdd_com, ann_ret_com = get_metrics(res_df["Combined_Nav"], res_df["Combined_Return"])
    cagr_us, mdd_us, _ = get_metrics(res_df["US_Nav"], res_df["US_Stock"])
    cagr_etf, mdd_etf, _ = get_metrics(res_df["ETF_Nav"], res_df["ETF_Rot"])
    cagr_bond, mdd_bond, _ = get_metrics(res_df["Bond_Nav"], res_df["Bond_Pred"])

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("🌐 綜合投資組合 (1/3配置)", f"年化報酬: {cagr_com:.2f}%", f"最大回落 (MDD): {mdd_com:.2f}%")
    mcol2.metric("🇺🇸 美股選股策略", f"年化報酬: {cagr_us:.2f}%", f"最大回落 (MDD): {mdd_us:.2f}%")
    mcol3.metric("🌐 跨資產 ETF 輪動", f"年化報酬: {cagr_etf:.2f}%", f"最大回落 (MDD): {mdd_etf:.2f}%")
    mcol4.metric("🏛️ 公債總經預測策略", f"年化報酬: {cagr_bond:.2f}%", f"最大回落 (MDD): {mdd_bond:.2f}%")

    st.markdown("---")
    st.markdown('<div class="section-header">📈 各策略與綜合組合累積淨值走勢圖 (NAV)</div>', unsafe_allow_html=True)
    
    chart_df = res_df[["Combined_Nav", "US_Nav", "ETF_Nav", "Bond_Nav"]].copy()
    chart_df.columns = ["綜合投資組合 (1/3配置)", "美股選股策略", "跨資產 ETF 輪動", "公債總經預測策略"]
    st.line_chart(chart_df)

    st.markdown("---")
    st.markdown('<div class="section-header">📅 每月綜合配置與各子策略報酬明細表</div>', unsafe_allow_html=True)
    display_tbl = res_df[["Combined_Return", "US_Stock", "ETF_Rot", "Bond_Pred"]].copy()
    display_tbl.columns = ["綜合組合報酬(%)", "美股選股報酬(%)", "ETF輪動報酬(%)", "公債策略報酬(%)"]
    display_tbl = display_tbl.round(2)
    st.dataframe(display_tbl, use_container_width=True)
