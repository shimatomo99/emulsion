import datetime
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import sklearn
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# データの保存先CSV
DATA_FILE = "dataset.csv"

# --- ページ全体のデザイン設定（ブルー＆クリーンな知的人気テーマ） ---
st.set_page_config(
    page_title="乳化安定性 予測AIアプリ", page_icon="🧪", layout="wide"
)

st.markdown(
    """
    <style>
    /* 全体の背景 */
    .stApp {
        background-color: #f4f7fb;
    }
    h1, h2, h3 {
        color: #1e3a8a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* 入力フォーム・セレクトボックスの背景を「白」くくっきりさせる */
    input, select, textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
        background-color: #ffffff !important;
        border-color: #94a3b8 !important;
    }
    
    label {
        color: #1e293b !important;
        font-weight: 500;
    }

    /* ボタンのカスタムデザイン（落ち着いたブルー） */
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    /* コンテナの角丸・背景 */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border-radius: 8px;
        border-color: #cbd5e1 !important;
        background-color: #ffffff;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title(
    "🧪 乳化安定性 予測AIアシスタント（編集タブ対応・完全版）"
)
st.markdown(
    "<p style='color: #475569; font-size: 1.1em;'>データ提出時と同様に、編集画面でも5℃・40℃のタブで全パラメータを管理できます。</p>",
    unsafe_allow_html=True,
)

# --- パスワード認証機能 ---
if "authenticated" not in st.session_state:
  st.session_state.authenticated = False

if not st.session_state.authenticated:
  st.warning("🔒 このアプリを利用するにはパスワードが必要です。")
  input_pass = st.text_input("パスワードを入力してください", type="password")

  if st.button("ログイン"):
    if input_pass == "Shimatomo7":
      st.session_state.authenticated = True
      st.success("ログインに成功しました！")
      st.rerun()
    else:
      st.error("パスワードが違います。")
  st.stop()

# --- 認証成功後のメインアプリ ---

# タブの切り替え
tab1, tab2, tab3, tab4 = st.tabs([
    "📁 データ提出・同時予測モード",
    "🔍 個別予測・確認モード",
    "📊 処方グループ別まとめ・編集",
    "📈 分析・統計（グラフ）",
])


# --- OpenCVを用いた高度な画像解析関数 ---
def extract_advanced_image_features(image_file):
  try:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_cv is None:
      return 0.0, 0.0, 0.0, 0.0, 0.0

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    particle_count = 0
    areas = []
    circularities = []

    for cnt in contours:
      area = cv2.contourArea(cnt)
      if 10 < area < 5000:
        particle_count += 1
        areas.append(area)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
          circularity = (4 * np.pi * area) / (perimeter**2)
          circularities.append(min(circularity, 1.0))

    mean_area = np.mean(areas) if areas else 0.0
    mean_circ = np.mean(circularities) if circularities else 0.0
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    image_file.seek(0)
    return (
        float(particle_count),
        float(mean_area),
        float(mean_circ),
        float(mean_brightness),
        float(std_brightness),
    )
  except Exception as e:
    st.error(f"画像解析エラー: {e}")
    image_file.seek(0)
    return 0.0, 0.0, 0.0, 0.0, 0.0


# --- 1. データ提出・同時予測タブ ---
with tab1:
  st.header("実験データの提出 & AI同時予測")
  st.markdown(
      "処方コード、共通の初期測定値（粒径・濁度）、および5℃・40℃それぞれの"
      "試験後データを入力して提出します。"
  )

  submit_date = st.date_input(
      "実験・提出日", datetime.date.today(), key="sub_date"
  )

  c_pres1, c_pres2 = st.columns([3, 1])
  with c_pres1:
    none_pres = st.checkbox("データなし", key="chk_sub_pres")
    if none_pres:
      prescription_no = "データなし"
      st.text_input(
          "処方コード", value="データなし", disabled=True, key="sub_prescription_dis"
      )
    else:
      prescription_no = st.text_input(
          "処方コード（例: A200204C）",
          value="A200204C",
          key="sub_prescription",
      )

  emulsifier = st.selectbox(
      "使用した乳化剤を選択",
      ["アラビアガム", "ポリグリ", "キラヤ"],
      key="sub_emulsifier",
  )

  st.subheader("💧 処方組成（水層・油層の量）の入力")
  col_w1, col_w2 = st.columns(2)
  with col_w1:
    none_water = st.checkbox("データなし", key="chk_sub_water")
    if none_water:
      water_amount = 0.0
      st.number_input(
          "水層の量 (g)", value=0.0, disabled=True, key="sub_water_dis"
      )
    else:
      water_amount = st.number_input(
          "水層の量 (g または mL)",
          min_value=0.0,
          max_value=10000.0,
          value=80.0,
          step=1.0,
          key="sub_water",
      )
  with col_w2:
    none_oil = st.checkbox("データなし", key="chk_sub_oil")
    if none_oil:
      oil_amount = 0.0
      st.number_input(
          "油層の量 (g)", value=0.0, disabled=True, key="sub_oil_dis"
      )
    else:
      oil_amount = st.number_input(
          "油層の量 (g または mL)",
          min_value=0.0,
          max_value=10000.0,
          value=20.0,
          step=1.0,
          key="sub_oil",
      )

  total_amount = water_amount + oil_amount
  oil_ratio = (
      (oil_amount / total_amount * 100.0) if total_amount > 0 else 0.0
  )
  st.info(
      f"📊 **自動計算された油分率**: {oil_ratio:.1f}% （総量:"
      f" {total_amount:.1f}g / 水層 {water_amount}g : 油層 {oil_amount}g）"
  )

  # --- タブの外（共通項目）：初期平均粒径 ＆ 虐待前（初期）濁度 ---
  st.divider()
  st.subheader("🔬 共通の初期測定値（虐待前）")
  st.markdown("すべての温度条件で共通となる製造直後の測定値です。")

  col_init1, col_init2 = st.columns(2)
  with col_init1:
    none_init = st.checkbox("初期平均粒径データなし", key="chk_sub_init_common")
    if none_init:
      initial_d50 = 0.0
      st.number_input(
          "初期平均粒径", value=0.0, disabled=True, key="sub_init_dis"
      )
    else:
      initial_d50 = st.number_input(
          "製造直後の初期 平均粒径 (nm)",
          min_value=0.0,
          max_value=10000.0,
          value=200.0,
          step=10.0,
          key="sub_init_d50",
      )

  with col_init2:
    none_turb_init = st.checkbox(
        "虐待前（初期）濁度データなし", key="chk_sub_turb_init_common"
    )
    if none_turb_init:
      initial_turbidity = 0.0
      st.number_input(
          "虐待前濁度", value=0.0, disabled=True, key="sub_turb_init_dis"
      )
    else:
      initial_turbidity = st.number_input(
          "虐待前（初期）濁度",
          min_value=0.0,
          max_value=10.0,
          value=1.0000,
          step=0.0001,
          format="%.4f",
          key="sub_turb_init",
      )

  st.divider()
  # --- CSV自動解析・ピーク積分機能の追加 ---
st.divider()
st.subheader("📊 粒度分布CSV自動解析・ピーク積分ツール（新規追加機能）")
st.markdown("粒度分布測定器のCSVファイルをアップロードすると、自動でグラフ化し、スライダーで範囲を指定して第1・第2ピークの積分値（面積比 %）を自動算出します。")

with st.expander("📁 CSVファイルのアップロード & ピーク解析ツールを開く", expanded=False):
    uploaded_csv = st.file_uploader("粒度分布CSVファイルを選択 (例: 粒径 vs 頻度)", type=["csv", "txt"], key="csv_peak_uploader")
    
    if uploaded_csv is not None:
        try:
            # エンコーディング自動判定またはフォールバック
            bytes_data = uploaded_csv.getvalue()
            df_csv = None
            for enc in ['utf-8', 'shift-jis', 'cp932', 'latin1']:
                try:
                    uploaded_csv.seek(0)
                    df_csv = pd.read_csv(uploaded_csv, encoding=enc)
                    break
                except Exception:
                    continue
            
            if df_csv is not None and not df_csv.empty:
                st.success("CSVファイルを正常に読み込みました！")
                st.dataframe(df_csv.head(5), use_container_width=True)
                
                # 列の自動推定（数値列を探索）
                numeric_cols = df_csv.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    col_x = st.selectbox("粒径データの列 (X軸: nm等)", numeric_cols, index=0, key="csv_col_x")
                    col_y = st.selectbox("頻度/強度の列 (Y軸: %等)", numeric_cols, index=1 if len(numeric_cols)>1 else 0, key="csv_col_y")
                    
                    x_data = df_csv[col_x].values
                    y_data = df_csv[col_y].values
                    
                    # グラフ描画
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(x_data, y_data, color="#1e3a8a", lw=2, label="粒度分布曲線")
                    ax.set_xlabel("Particle Size (nm)")
                    ax.set_ylabel("Frequency (%)")
                    ax.set_title("Particle Size Distribution Curve")
                    ax.grid(True, linestyle="--", alpha=0.6)
                    
                    # ピーク範囲スライダー
                    xmin, xmax = float(np.min(x_data)), float(np.max(x_data))
                    st.markdown("#### 🎛️ ピーク範囲の指定（スライダー）")
                    col_sl1, col_sl2 = st.columns(2)
                    with col_sl1:
                        p1_range = st.slider("第1ピーク 範囲 (nm)", xmin, xmax, (xmin, min(xmin + (xmax-xmin)*0.5, xmax)), key="slider_p1")
                    with col_sl2:
                        p2_range = st.slider("第2ピーク 範囲 (nm)", xmin, xmax, (min(xmin + (xmax-xmin)*0.5, xmax), xmax), key="slider_p2")
                    
                    # 積分値（面積比）の計算
                    mask_p1 = (x_data >= p1_range[0]) & (x_data <= p1_range[1])
                    mask_p2 = (x_data >= p2_range[0]) & (x_data <= p2_range[1])
                    
                    area_total = np.trapz(y_data, x_data) if np.trapz(y_data, x_data) != 0 else 1.0
                    area_p1 = np.trapz(y_data[mask_p1], x_data[mask_p1]) if np.sum(mask_p1) > 1 else 0.0
                    area_p2 = np.trapz(y_data[mask_p2], x_data[mask_p2]) if np.sum(mask_p2) > 1 else 0.0
                    
                    ratio_p1 = (area_p1 / area_total * 100.0) if area_total > 0 else 0.0
                    ratio_p2 = (area_p2 / area_total * 100.0) if area_total > 0 else 0.0
                    
                    # グラフに範囲をハイライト
                    ax.fill_between(x_data, y_data, where=mask_p1, color='blue', alpha=0.3, label=f"第1ピーク 面積比: {ratio_p1:.1f}%")
                    ax.fill_between(x_data, y_data, where=mask_p2, color='orange', alpha=0.3, label=f"第2ピーク 面積比: {ratio_p2:.1f}%")
                    ax.legend()
                    st.pyplot(fig)
                    
                    st.info(f"📊 **自動算出された積分値（面積比）**：\n- 第1ピーク ({p1_range[0]:.1f} ~ {p1_range[1]:.1f} nm): **{ratio_p1:.1f}%**\n- 第2ピーク ({p2_range[0]:.1f} ~ {p2_range[1]:.1f} nm): **{ratio_p2:.1f}%**")
                    
                    if st.button("✨ この計算結果を以下の入力フォーム（5℃/40℃）に反映する"):
                        st.session_state["auto_p1_min"] = p1_range[0]
                        st.session_state["auto_p1_max"] = p1_range[1]
                        st.session_state["auto_p1_rat"] = ratio_p1
                        st.session_state["auto_p2_min"] = p2_range[0]
                        st.session_state["auto_p2_max"] = p2_range[1]
                        st.session_state["auto_p2_rat"] = ratio_p2
                        st.success("ピーク範囲と面積比をセッションに反映しました！下の入力フォームをご確認ください。")
                else:
                    st.warning("CSVファイルに数値データ列が2つ以上見つかりませんでした。")
            else:
                st.error("CSVファイルの読み込みに失敗しました。")
        except Exception as e:
            st.error(f"CSV解析エラー: {e}")

st.divider()
st.subheader("🌡️ 温度別（5℃ / 40℃）の試験後データ入力タブ")
  st.markdown(
      "それぞれの温度における試験後粒径・濁度・ピーク範囲・顕微鏡写真を"
      "入力してください。"
  )

  # --- 5℃と40℃の入力用サブタブ ---
  sub_tab_5c, sub_tab_40c = st.tabs(["❄️ 5℃ 試験データ", "🔥 40℃ 試験データ"])

  # --- 5℃データの入力 ---
  with sub_tab_5c:
    st.markdown("### ❄️ 5℃ 試験後データ")
    col_c1_5, col_c2_5 = st.columns(2)
    with col_c1_5:
      test_days_5 = st.number_input(
          "5℃試験 期間 (日)",
          min_value=1,
          max_value=365,
          value=14,
          step=1,
          key="sub_days_5",
      )
    with col_c2_5:
      none_after_5 = st.checkbox("5℃試験後粒径データなし", key="chk_after_5")
      if none_after_5:
        after_d50_5 = 0.0
        st.number_input(
            "5℃試験後粒径", value=0.0, disabled=True, key="after_d50_5_dis"
        )
      else:
        after_d50_5 = st.number_input(
            "5℃試験後 平均粒径 (nm)",
            min_value=0.0,
            max_value=10000.0,
            value=205.0,
            step=10.0,
            key="after_d50_5",
        )

    delta_d50_5 = (
        after_d50_5 - initial_d50
        if not none_after_5 and not none_init
        else 0.0
    )

    st.markdown("#### 💧 5℃ 試験後濁度")
    none_turb_5 = st.checkbox("5℃試験後濁度：データなし", key="chk_turb_5")
    if none_turb_5:
      turbidity_5 = 0.0
      st.text("（濁度: なし）")
    else:
      turbidity_5 = st.number_input(
          "5℃ 試験後濁度",
          value=1.0000,
          step=0.0001,
          format="%.4f",
          key="turb_5",
      )

    st.markdown("#### 🔬 5℃ レーザー解析（ピーク範囲）")
    col_l1_5, col_l2_5 = st.columns(2)
    with col_l1_5:
      none_p1_5 = st.checkbox("5℃ 第1ピーク：データなし", key="chk_p1_5")
      if none_p1_5:
        peak1_min_5, peak1_max_5, peak1_ratio_5 = 0.0, 0.0, 0.0
        st.text("（第1ピーク: なし）")
      else:
        peak1_min_5 = st.number_input(
            "5℃ 第1ピーク 最小値 (nm)",
            value=50.0,
            step=10.0,
            key="p1_min_5",
        )
        peak1_max_5 = st.number_input(
            "5℃ 第1ピーク 最大値 (nm)",
            value=500.0,
            step=10.0,
            key="p1_max_5",
        )
        peak1_ratio_5 = st.number_input(
            "5℃ 第1ピーク 面積比 (%)",
            value=95.0,
            step=0.5,
            format="%.1f",
            key="p1_rat_5",
        )
    with col_l2_5:
      none_p2_5 = st.checkbox("5℃ 第2ピーク：データなし", key="chk_p2_5")
      if none_p2_5:
        peak2_min_5, peak2_max_5, peak2_ratio_5 = 0.0, 0.0, 0.0
        st.text("（第2ピーク: なし）")
      else:
        peak2_min_5 = st.number_input(
            "5℃ 第2ピーク 最小値 (nm)",
            value=500.0,
            step=10.0,
            key="p2_min_5",
        )
        peak2_max_5 = st.number_input(
            "5℃ 第2ピーク 最大値 (nm)",
            value=5000.0,
            step=10.0,
            key="p2_max_5",
        )
        peak2_ratio_5 = st.number_input(
            "5℃ 第2ピーク 面積比 (%)",
            value=5.0,
            step=0.5,
            format="%.1f",
            key="p2_rat_5",
        )

    st.markdown("#### 📸 5℃条件の顕微鏡写真")
    file_5 = st.file_uploader(
        "顕微鏡写真ファイルをアップロード",
        type=["png", "jpg", "jpeg"],
        key="file_5",
    )

  # --- 40℃データの入力 ---
  with sub_tab_40c:
    st.markdown("### 🔥 40℃ 試験後データ")
    col_c1_40, col_c2_40 = st.columns(2)
    with col_c1_40:
      test_days_40 = st.number_input(
          "40℃試験 期間 (日)",
          min_value=1,
          max_value=365,
          value=14,
          step=1,
          key="sub_days_40",
      )
    with col_c2_40:
      none_after_40 = st.checkbox("40℃試験後粒径データなし", key="chk_after_40")
      if none_after_40:
        after_d50_40 = 0.0
        st.number_input(
            "40℃試験後粒径", value=0.0, disabled=True, key="after_d50_40_dis"
        )
      else:
        after_d50_40 = st.number_input(
            "40℃試験後 平均粒径 (nm)",
            min_value=0.0,
            max_value=10000.0,
            value=220.0,
            step=10.0,
            key="after_d50_40",
        )

    delta_d50_40 = (
        after_d50_40 - initial_d50
        if not none_after_40 and not none_init
        else 0.0
    )

    st.markdown("#### 💧 40℃ 試験後濁度")
    none_turb_40 = st.checkbox("40℃試験後濁度：データなし", key="chk_turb_40")
    if none_turb_40:
      turbidity_40 = 0.0
      st.text("（濁度: なし）")
    else:
      turbidity_40 = st.number_input(
          "40℃ 試験後濁度",
          value=1.0000,
          step=0.0001,
          format="%.4f",
          key="turb_40",
      )

    st.markdown("#### 🔬 40℃ レーザー解析（ピーク範囲）")
    col_l1_40, col_l2_40 = st.columns(2)
    with col_l1_40:
      none_p1_40 = st.checkbox("40℃ 第1ピーク：データなし", key="chk_p1_40")
      if none_p1_40:
        peak1_min_40, peak1_max_40, peak1_ratio_40 = 0.0, 0.0, 0.0
        st.text("（第1ピーク: なし）")
      else:
        peak1_min_40 = st.number_input(
            "40℃ 第1ピーク 最小値 (nm)",
            value=50.0,
            step=10.0,
            key="p1_min_40",
        )
        peak1_max_40 = st.number_input(
            "40℃ 第1ピーク 最大値 (nm)",
            value=500.0,
            step=10.0,
            key="p1_max_40",
        )
        peak1_ratio_40 = st.number_input(
            "40℃ 第1ピーク 面積比 (%)",
            value=95.0,
            step=0.5,
            format="%.1f",
            key="p1_rat_40",
        )
    with col_l2_40:
      none_p2_40 = st.checkbox("40℃ 第2ピーク：データなし", key="chk_p2_40")
      if none_p2_40:
        peak2_min_40, peak2_max_40, peak2_ratio_40 = 0.0, 0.0, 0.0
        st.text("（第2ピーク: なし）")
      else:
        peak2_min_40 = st.number_input(
            "40℃ 第2ピーク 最小値 (nm)",
            value=500.0,
            step=10.0,
            key="p2_min_40",
        )
        peak2_max_40 = st.number_input(
            "40℃ 第2ピーク 最大値 (nm)",
            value=5000.0,
            step=10.0,
            key="p2_max_40",
        )
        peak2_ratio_40 = st.number_input(
            "40℃ 第2ピーク 面積比 (%)",
            value=5.0,
            step=0.5,
            format="%.1f",
            key="p2_rat_40",
        )

    st.markdown("#### 📸 40℃条件の顕微鏡写真")
    file_40 = st.file_uploader(
        "顕微鏡写真ファイルをアップロード",
        type=["png", "jpg", "jpeg"],
        key="file_40",
    )

  st.divider()
  condition_memo = st.text_input(
      "共通メモ（例: ホモジナイザー条件、遠心分離など）",
      value="",
      key="sub_condition",
  )

  stability = st.radio(
      "9か月以内の全体安定性判定（教師データ用）",
      ["安定（長期間分離しない）", "不安定（9か月以内に分離・クリーミング）"],
      key="sub_stability",
  )

  unstable_days = 270
  if stability == "不安定（9か月以内に分離・クリーミング）":
    unstable_days = st.number_input(
        "何日で不安定になりましたか？（日数）",
        min_value=1,
        max_value=270,
        value=30,
        key="sub_unstable_days",
    )

  # --- データ提出ボタン処理 ---
  if st.button("データ提出 ＆ AI同時予測を実行"):
    chosen_file = file_40 if file_40 is not None else file_5

    if chosen_file is not None:
      folder_map = {
          "アラビアガム": "data/arabic_gum",
          "ポリグリ": "data/polygly",
          "キラヤ": "data/quillaja",
      }
      save_dir = folder_map[emulsifier]
      os.makedirs(save_dir, exist_ok=True)

      image_path = os.path.join(save_dir, chosen_file.name)
      img_bytes_for_save = chosen_file.getvalue()
      with open(image_path, "wb") as f:
        f.write(img_bytes_for_save)

      chosen_file.seek(0)
      p_count, m_area, m_circ, m_bright, s_bright = (
          extract_advanced_image_features(chosen_file)
      )

      is_stable_val = 1 if stability.startswith("安定") else 0

      new_data = pd.DataFrame([{
          "date": str(submit_date),
          "prescription_no": (
              str(prescription_no) if prescription_no else "未設定"
          ),
          "emulsifier": emulsifier,
          "water_amount": water_amount,
          "oil_amount": oil_amount,
          "oil_ratio": oil_ratio,
          "temperature": 40.0,
          "test_days": test_days_40,
          "initial_d50": initial_d50,
          "initial_turbidity": (
              initial_turbidity if not none_turb_init else 0.0
          ),
          "after_d50": after_d50_40,
          "delta_d50": delta_d50_40,
          "peak1_min": peak1_min_40,
          "peak1_max": peak1_max_40,
          "peak1_ratio": peak1_ratio_40,
          "peak2_min": peak2_min_40,
          "peak2_max": peak2_max_40,
          "peak2_ratio": peak2_ratio_40,
          "initial_d50_5": initial_d50,
          "after_d50_5": after_d50_5,
          "delta_d50_5": delta_d50_5,
          "peak1_ratio_5": peak1_ratio_5,
          "peak2_ratio_5": peak2_ratio_5,
          "turbidity_5": turbidity_5,
          "particle_size": after_d50_40,
          "turbidity": turbidity_40,
          "condition": str(condition_memo) if condition_memo else "",
          "stability_text": stability,
          "is_stable": is_stable_val,
          "days": unstable_days,
          "image_path": image_path,
          "particle_count": p_count,
          "mean_area": m_area,
          "circularity": m_circ,
          "img_mean": m_bright,
          "img_std": s_bright,
      }])

      if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
        try:
          df = pd.read_csv(DATA_FILE)
          df = pd.concat([df, new_data], ignore_index=True)
        except Exception:
          df = new_data
      else:
        df = new_data

      df["prescription_no"] = df["prescription_no"].astype(str)
      df["condition"] = df["condition"].astype(str)
      df.to_csv(DATA_FILE, index=False)
      st.success("測定データを正常に保存しました！")

      # --- 提出と同時にAI予測を実行 ---
      st.divider()
      st.header("🤖 同時AI予測・診断結果（40℃加速試験基準）")

      train_df = df[df["emulsifier"] == emulsifier]
      if len(train_df) < 2:
        train_df = df

      if len(train_df) < 2:
        st.info(
            "※学習データが2件未満のため、AI予測を行うにはさらにデータを蓄積する"
            "必要があります。"
        )
      else:
        base_feature_cols = [
            "oil_ratio",
            "water_amount",
            "oil_amount",
            "delta_d50",
            "initial_d50",
            "after_d50",
            "peak1_min",
            "peak1_max",
            "peak1_ratio",
            "peak2_min",
            "peak2_max",
            "peak2_ratio",
            "particle_count",
            "mean_area",
            "circularity",
            "img_mean",
            "img_std",
            "temperature",
            "test_days",
            "turbidity",
        ]

        val_map = {
            "oil_ratio": oil_ratio,
            "water_amount": 0.0 if none_water else water_amount,
            "oil_amount": 0.0 if none_oil else oil_amount,
            "delta_d50": delta_d50_40,
            "initial_d50": 0.0 if none_init else initial_d50,
            "after_d50": 0.0 if none_after_40 else after_d50_40,
            "peak1_min": 0.0 if none_p1_40 else peak1_min_40,
            "peak1_max": 0.0 if none_p1_40 else peak1_max_40,
            "peak1_ratio": 0.0 if none_p1_40 else peak1_ratio_40,
            "peak2_min": 0.0 if none_p2_40 else peak2_min_40,
            "peak2_max": 0.0 if none_p2_40 else peak2_max_40,
            "peak2_ratio": 0.0 if none_p2_40 else peak2_ratio_40,
            "temperature": 40.0,
            "test_days": test_days_40,
            "turbidity": 0.0 if none_turb_40 else turbidity_40,
            "particle_count": p_count,
            "mean_area": m_area,
            "circularity": m_circ,
            "img_mean": m_bright,
            "img_std": s_bright,
        }

        active_feature_cols = []
        for col in base_feature_cols:
          if col in train_df.columns:
            if val_map[col] != 0.0 or col in [
                "temperature",
                "test_days",
                "oil_ratio",
            ]:
              active_feature_cols.append(col)

        if len(active_feature_cols) < 2:
          active_feature_cols = ["temperature", "test_days", "oil_ratio"]

        for col in active_feature_cols:
          if col not in train_df.columns:
            train_df[col] = 0.0

        train_clean = train_df.dropna(subset=active_feature_cols + ["is_stable"])

        if len(train_clean) < 2:
          train_clean = train_df.dropna(subset=["is_stable"])
          active_feature_cols = [
              c
              for c in ["temperature", "test_days", "oil_ratio"]
              if c in train_clean.columns
          ]

        X = train_clean[active_feature_cols]
        y_class = train_clean["is_stable"]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y_class)

        X_pred = np.array([[val_map[col] for col in active_feature_cols]])

        prediction = model.predict(X_pred)[0]
        proba = model.predict_proba(X_pred)[0]
        confidence = max(proba) * 100

        if prediction == 1:
          st.success(f"判定：**9か月間安定**（予測確信度: {confidence:.1f}%）")
          st.balloons()
        else:
          unstable_subset = train_clean[train_clean["is_stable"] == 0]
          avg_unstable_days = (
              int(unstable_subset["days"].mean())
              if len(unstable_subset) > 0 and "days" in unstable_subset
              else 30
          )

          st.error(
              f"🚨 【リスクアラート発令】 9か月以内に不安定化・分離の恐れあり"
              f"（予測確信度: {confidence:.1f}%）"
          )
          st.warning(
              f"⚠️ **寿命予測の注意喚起**: 有効な測定データ傾向から推測して、約"
              f" **{avg_unstable_days} 日前後** で破綻する恐れがあります。"
          )

          st.info("🤖 **AIからの改善ヒント・アドバイス**")
          advice_list = []
          if not none_oil and oil_ratio > 40.0:
            advice_list.append(
                "・**油分率が高めの処方です**：乳化剤の増量を推奨します。"
            )
          if not none_p2_40 and peak2_ratio_40 > 3.0:
            advice_list.append("・**40℃第2ピーク比率が高めです**。")
          if not none_after_40 and not none_init and delta_d50_40 > 30:
            advice_list.append("・**40℃粒径変動スピードが大きい**です。")
          if not advice_list:
            advice_list.append("・処方の再検討をおすすめします。")

          for advice in advice_list:
            st.write(advice)
    else:
      st.warning("顕微鏡写真を少なくとも1枚アップロードしてください。")

# --- 2. 個別予測・確認タブ ---
with tab2:
  st.header("個別AI予測・確認モード")
  st.markdown("提出データとは別に、仮想条件でAI予測したい場合にご利用ください。")

  pred_emulsifier = st.selectbox(
      "評価する乳化剤を選択",
      ["アラビアガム", "ポリグリ", "キラヤ"],
      key="pred_emulsifier",
  )

  col_pw1, col_pw2 = st.columns(2)
  with col_pw1:
    pred_water = st.number_input(
        "予測時の水層の量 (g または mL)", value=80.0, key="pred_water"
    )
  with col_pw2:
    pred_oil = st.number_input(
        "予測時の油層の量 (g または mL)", value=20.0, key="pred_oil"
    )

  pred_total = pred_water + pred_oil
  pred_oil_ratio = (
      (pred_oil / pred_total * 100.0) if pred_total > 0 else 0.0
  )
  st.info(
      f"📊 **予測時 油分率**: {pred_oil_ratio:.1f}% （総量: {pred_total:.1f}g）"
  )

  pred_init_d50 = st.number_input(
      "予測時の初期平均粒径 (nm)", value=200.0, key="pred_init_d50"
  )
  pred_after_d50 = st.number_input(
      "予測時の試験後平均粒径 (nm)", value=220.0, key="pred_after_d50"
  )
  pred_delta_d50 = pred_after_d50 - pred_init_d50

  pred_file = st.file_uploader(
      "📸 確認用コロイドの顕微鏡写真をアップロード",
      type=["png", "jpg", "jpeg"],
      key="pred_file",
  )

  if st.button("単独予測を実行"):
    if pred_file is not None:
      if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        st.warning("学習データがありません。")
      else:
        df = pd.read_csv(DATA_FILE)
        train_df = df[df["emulsifier"] == pred_emulsifier]
        if len(train_df) < 2:
          train_df = df

        feature_cols = [
            "oil_ratio",
            "water_amount",
            "oil_amount",
            "delta_d50",
            "initial_d50",
            "after_d50",
            "temperature",
        ]
        for col in feature_cols:
          if col not in train_df.columns:
            train_df[col] = 0.0

        train_clean = train_df.dropna(subset=feature_cols + ["is_stable"])
        X = train_clean[feature_cols]
        y_class = train_clean["is_stable"]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y_class)

        p_count, m_area, m_circ, p_mean, p_std = (
            extract_advanced_image_features(pred_file)
        )
        X_pred = np.array([[
            pred_oil_ratio,
            pred_water,
            pred_oil,
            pred_delta_d50,
            pred_init_d50,
            pred_after_d50,
            40.0,
        ]])

        prediction = model.predict(X_pred)[0]
        proba = model.predict_proba(X_pred)[0]
        confidence = max(proba) * 100

        if prediction == 1:
          st.success(f"判定：**9か月間安定**（予測確信度: {confidence:.1f}%）")
        else:
          st.error(
              f"🚨 【リスクアラート発令】 9か月以内に分離の恐れあり（予測確信度:"
              f" {confidence:.1f}%）"
          )
    else:
      st.warning("写真をアップロードしてください。")

# --- 3. 処方グループ別まとめ・編集タブ ---
with tab3:
  st.header("📁 処方グループ別データまとめ・編集")
  st.markdown(
      "ベースコードが共通するデータをまとめ、日付順または名前順で並び替えて"
      "確認・編集できます。"
  )

  if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
    st.write("まだ提出されたデータはありません。")
  else:
    df = pd.read_csv(DATA_FILE)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 全データをCSV形式でダウンロード",
        data=csv_data,
        file_name="emulsion_dataset_export.csv",
        mime="text/csv",
    )
    st.divider()

    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
      search_query = st.text_input(
          "🔍 処方コードで検索（例: A200204C など）", "", key="hist_search_pres"
      )
    with col_s2:
      sort_target = st.selectbox(
          "🔄 並び替え基準",
          ["日付順", "名前順（処方コード）"],
          key="hist_sort_target",
      )
    with col_s3:
      if sort_target == "日付順":
        sort_order = st.selectbox(
            "📅 順序",
            ["降順（新しい順）", "昇順（古い順）"],
            key="hist_sort_order_date",
        )
      else:
        sort_order = st.selectbox(
            "🔤 順序",
            ["昇順（A→Z / 50音順）", "降順（Z→A）"],
            key="hist_sort_order_name",
        )

    df_display = df.copy()
    if search_query:
      df_display["prescription_no"] = (
          df_display["prescription_no"].astype(str).fillna("")
      )
      df_display = df_display[
          df_display["prescription_no"].str.contains(search_query, case=False)
      ]

    if len(df_display) == 0:
      st.write("該当するデータはありません。")
    else:
      if "prescription_no" not in df_display.columns:
        df_display["prescription_no"] = "未設定"

      base_groups = []
      for p_no in df_display["prescription_no"].astype(str):
        base = p_no.split("-")[0].split(" ")[0].strip()
        base_groups.append(base if base else "未設定")

      df_display["base_prescription"] = base_groups

      unique_name_counts = {}
      assigned_group_names = []

      for idx, row in df_display.iterrows():
        p_no = str(row["prescription_no"])
        if p_no not in unique_name_counts:
          unique_name_counts[p_no] = 0

        unique_name_counts[p_no] += 1
        count = unique_name_counts[p_no]

        if count == 1:
          assigned_group_names.append(p_no)
        else:
          assigned_group_names.append(f"{p_no}-{count}")

      df_display["display_group_name"] = assigned_group_names

      if sort_target == "日付順":
        if "date" in df_display.columns:
          df_display["date_parsed"] = pd.to_datetime(
              df_display["date"], errors="coerce"
          )
          is_ascending = True if "古い順" in sort_order else False
          df_display = df_display.sort_values(
              by=["date_parsed", "date"], ascending=is_ascending
          )
      else:
        is_ascending = True if "昇順" in sort_order else False
        df_display = df_display.sort_values(
            by=["display_group_name"], ascending=is_ascending
        )

      grouped = df_display.groupby("display_group_name", sort=False)

      st.write(
          f"グループ数: **{len(grouped)}種類** （総データ数: {len(df_display)}件）"
      )

      for group_name, group_df in grouped:
        with st.expander(
            f"🏷️ 処方グループ: {group_name} （データ件数: {len(group_df)}件）",
            expanded=True,
        ):
          for index, row in group_df.iterrows():
            with st.container(border=True):
              col1, col2, col3 = st.columns([1.2, 2.8, 0.8])

              with col1:
                img_path = str(row.get("image_path", ""))
                if os.path.exists(img_path):
                  image = Image.open(img_path)
                  st.image(
                      image,
                      caption=f"提出日: {row.get('date', '不明')}",
                      width=140,
                  )
                else:
                  st.write("（写真なし）")

              with col2:
                st.markdown(
                    f"### 🧪 登録名/条件: {row.get('prescription_no', '')} |"
                    f" 乳化剤: {row.get('emulsifier', '不明')}"
                )

                w_amt = row.get("water_amount", "-")
                o_amt = row.get("oil_amount", "-")
                o_rat = row.get("oil_ratio", 0.0)
                d50_init = row.get("initial_d50", "-")
                turb_init = row.get("initial_turbidity", "-")
                d50_after = row.get("after_d50", "-")
                d_d50 = row.get("delta_d50", 0)

                d50_after_5 = row.get("after_d50_5", "-")

                st.write(
                    f"📅 **提出日**: {row.get('date', '不明')}  \n"
                    f"💧 **処方組成**: 水層 {w_amt}g / 油層 {o_amt}g （油分率:"
                    f" {float(o_rat):.1f}%）  \n"
                    f"🔬 **共通初期値**: 初期粒径 {d50_init}nm / 虐待前濁度"
                    f" {turb_init}  \n"
                    f"🔥 **40℃試験後**: {d50_after}nm (Δ: {float(d_d50):+.1f}nm)"
                    f"  \n"
                    f"❄️ **5℃試験後**: {d50_after_5}nm  \n"
                    f"⚖️ **判定**: {row.get('stability_text', '')}"
                )
                if "不安定" in str(row.get("stability_text", "")):
                  st.write(
                      f"⏱️ **不安定化までの日数**: {row.get('days', '')} 日"
                  )
                if row.get("condition"):
                  st.write(f"📝 **メモ**: {row.get('condition')}")

              with col3:
                with st.popover("🖥️ 大画面編集"):
                  st.write(
                      f"### ✏️ データの全項目編集 (ID: {index} / 処方:"
                      f" {row.get('prescription_no', '')})"
                  )

                  edit_pres = st.text_input(
                      "処方コード",
                      value=str(row.get("prescription_no", "")),
                      key=f"e_pres_{index}",
                  )
                  current_emulsifier = str(
                      row.get("emulsifier", "アラビアガム")
                  )
                  emulsifier_idx = (
                      ["アラビアガム", "ポリグリ", "キラヤ"].index(
                          current_emulsifier
                      )
                      if current_emulsifier
                      in ["アラビアガム", "ポリグリ", "キラヤ"]
                      else 0
                  )
                  edit_emulsifier = st.selectbox(
                      "使用した乳化剤",
                      ["アラビアガム", "ポリグリ", "キラヤ"],
                      index=emulsifier_idx,
                      key=f"e_emul_{index}",
                  )

                  st.markdown("#### 💧 処方組成の編集")
                  col_ew1, col_ew2 = st.columns(2)
                  with col_ew1:
                    edit_water = st.number_input(
                        "水層の量 (g)",
                        value=float(row.get("water_amount", 80.0)),
                        key=f"e_water_{index}",
                    )
                  with col_ew2:
                    edit_oil = st.number_input(
                        "油層の量 (g)",
                        value=float(row.get("oil_amount", 20.0)),
                        key=f"e_oil_{index}",
                    )

                  st.markdown("#### 🔬 共通初期値の編集（虐待前）")
                  col_ei1, col_ei2 = st.columns(2)
                  with col_ei1:
                    edit_init = st.number_input(
                        "初期平均粒径 (nm)",
                        value=float(row.get("initial_d50", 200.0)),
                        key=f"e_init_{index}",
                    )
                  with col_ei2:
                    edit_turb_init = st.number_input(
                        "虐待前（初期）濁度",
                        value=float(
                            row.get("initial_turbidity", 1.0)
                            if pd.notnull(row.get("initial_turbidity"))
                            else 1.0
                        ),
                        format="%.4f",
                        key=f"e_turbinit_{index}",
                    )

                  st.markdown(
                      "#### 🌡️ 温度別（5℃ /"
                      " 40℃）の試験後データ編集タブ"
                  )
                  edit_tab_5, edit_tab_40 = st.tabs(
                      ["❄️ 5℃ 試験データ", "🔥 40℃ 試験データ"]
                  )

                  with edit_tab_5:
                    edit_test_days_5 = st.number_input(
                        "5℃試験 期間 (日)",
                        value=int(
                            row.get("test_days", 14)
                            if pd.notnull(row.get("test_days"))
                            else 14
                        ),
                        key=f"e_days_5_{index}",
                    )
                    edit_after_5 = st.number_input(
                        "5℃試験後平均粒径 (nm)",
                        value=float(
                            row.get("after_d50_5", 205.0)
                            if pd.notnull(row.get("after_d50_5"))
                            else 205.0
                        ),
                        key=f"e_after5_{index}",
                    )
                    edit_turb_5 = st.number_input(
                        "5℃ 試験後濁度",
                        value=float(
                            row.get("turbidity_5", 1.0)
                            if pd.notnull(row.get("turbidity_5"))
                            else 1.0
                        ),
                        format="%.4f",
                        key=f"e_turb5_{index}",
                    )
                    edit_p1_min_5 = st.number_input(
                        "5℃ 第1ピーク 最小値",
                        value=float(row.get("peak1_min", 50.0)),
                        key=f"e_p1min5_{index}",
                    )
                    edit_p1_max_5 = st.number_input(
                        "5℃ 第1ピーク 最大値",
                        value=float(row.get("peak1_max", 500.0)),
                        key=f"e_p1max5_{index}",
                    )
                    edit_p1_ratio_5 = st.number_input(
                        "5℃ 第1ピーク比率 (%)",
                        value=float(
                            row.get("peak1_ratio_5", row.get("peak1_ratio", 95.0))
                        ),
                        key=f"e_p1rat5_{index}",
                    )
                    edit_p2_min_5 = st.number_input(
                        "5℃ 第2ピーク 最小値",
                        value=float(row.get("peak2_min", 500.0)),
                        key=f"e_p2min5_{index}",
                    )
                    edit_p2_max_5 = st.number_input(
                        "5℃ 第2ピーク 最大値",
                        value=float(row.get("peak2_max", 5000.0)),
                        key=f"e_p2max5_{index}",
                    )
                    edit_p2_ratio_5 = st.number_input(
                        "5℃ 第2ピーク比率 (%)",
                        value=float(
                            row.get("peak2_ratio_5", row.get("peak2_ratio", 5.0))
                        ),
                        key=f"e_p2rat5_{index}",
                    )

                  with edit_tab_40:
                    edit_test_days_40 = st.number_input(
                        "40℃試験 期間 (日)",
                        value=int(
                            row.get("test_days", 14)
                            if pd.notnull(row.get("test_days"))
                            else 14
                        ),
                        key=f"e_days_40_{index}",
                    )
                    edit_after = st.number_input(
                        "40℃試験後平均粒径 (nm)",
                        value=float(row.get("after_d50", 220.0)),
                        key=f"e_after_{index}",
                    )
                    edit_turb_40 = st.number_input(
                        "40℃ 試験後濁度",
                        value=float(
                            row.get("turbidity", 1.0)
                            if pd.notnull(row.get("turbidity"))
                            else 1.0
                        ),
                        format="%.4f",
                        key=f"e_turb40_{index}",
                    )
                    edit_p1_min_40 = st.number_input(
                        "40℃ 第1ピーク 最小値",
                        value=float(row.get("peak1_min", 50.0)),
                        key=f"e_p1min40_{index}",
                    )
                    edit_p1_max_40 = st.number_input(
                        "40℃ 第1ピーク 最大値",
                        value=float(row.get("peak1_max", 500.0)),
                        key=f"e_p1max40_{index}",
                    )
                    edit_p1_ratio_40 = st.number_input(
                        "40℃ 第1ピーク比率 (%)",
                        value=float(row.get("peak1_ratio", 95.0)),
                        key=f"e_p1rat40_{index}",
                    )
                    edit_p2_min_40 = st.number_input(
                        "40℃ 第2ピーク 最小値",
                        value=float(row.get("peak2_min", 500.0)),
                        key=f"e_p2min40_{index}",
                    )
                    edit_p2_max_40 = st.number_input(
                        "40℃ 第2ピーク 最大値",
                        value=float(row.get("peak2_max", 5000.0)),
                        key=f"e_p2max40_{index}",
                    )
                    edit_p2_ratio_40 = st.number_input(
                        "40℃ 第2ピーク比率 (%)",
                        value=float(row.get("peak2_ratio", 5.0)),
                        key=f"e_p2rat40_{index}",
                    )

                  st.markdown("---")
                  edit_cond = st.text_input(
                      "その他メモ",
                      value=(
                          str(row.get("condition", ""))
                          if pd.notnull(row.get("condition"))
                          else ""
                      ),
                      key=f"e_cond_{index}",
                  )

                  cur_stab = str(row.get("stability_text", "安定"))
                  edit_stability = st.radio(
                      "9か月以内の安定性",
                      [
                          "安定（長期間分離しない）",
                          "不安定（9か月以内に分離・クリーミング）",
                      ],
                      index=0 if cur_stab.startswith("安定") else 1,
                      key=f"e_stab_{index}",
                  )

                  if st.button(
                      "💾 すべての変更を保存する", key=f"save_edit_{index}"
                  ):
                    df["prescription_no"] = df["prescription_no"].astype(str)
                    df["condition"] = df["condition"].astype(str)

                    tot = edit_water + edit_oil
                    new_o_rat = (edit_oil / tot * 100.0) if tot > 0 else 0.0
                    new_delta_40 = edit_after - edit_init
                    new_delta_5 = edit_after_5 - edit_init
                    new_is_stable = (
                        1 if edit_stability.startswith("安定") else 0
                    )

                    df.loc[index, "prescription_no"] = str(edit_pres)
                    df.loc[index, "emulsifier"] = edit_emulsifier
                    df.loc[index, "water_amount"] = float(edit_water)
                    df.loc[index, "oil_amount"] = float(edit_oil)
                    df.loc[index, "oil_ratio"] = float(new_o_rat)
                    df.loc[index, "initial_d50"] = float(edit_init)
                    df.loc[index, "initial_turbidity"] = float(edit_turb_init)

                    # 40℃データの反映
                    df.loc[index, "test_days"] = int(edit_test_days_40)
                    df.loc[index, "after_d50"] = float(edit_after)
                    df.loc[index, "delta_d50"] = float(new_delta_40)
                    df.loc[index, "turbidity"] = float(edit_turb_40)
                    df.loc[index, "peak1_min"] = float(edit_p1_min_40)
                    df.loc[index, "peak1_max"] = float(edit_p1_max_40)
                    df.loc[index, "peak1_ratio"] = float(edit_p1_ratio_40)
                    df.loc[index, "peak2_min"] = float(edit_p2_min_40)
                    df.loc[index, "peak2_max"] = float(edit_p2_max_40)
                    df.loc[index, "peak2_ratio"] = float(edit_p2_ratio_40)

                    # 5℃データの反映
                    df.loc[index, "after_d50_5"] = float(edit_after_5)
                    df.loc[index, "delta_d50_5"] = float(new_delta_5)
                    df.loc[index, "turbidity_5"] = float(edit_turb_5)
                    df.loc[index, "peak1_ratio_5"] = float(edit_p1_ratio_5)
                    df.loc[index, "peak2_ratio_5"] = float(edit_p2_ratio_5)

                    df.loc[index, "condition"] = str(edit_cond)
                    df.loc[index, "stability_text"] = edit_stability
                    df.loc[index, "is_stable"] = int(new_is_stable)

                    df.to_csv(DATA_FILE, index=False)
                    st.success("すべての変更を正常に保存しました！")
                    st.rerun()

                st.write("")
                with st.popover("✕ 削除"):
                  st.write("このデータを削除しますか？")
                  if st.button("はい、削除する", key=f"confirm_del_{index}"):
                    df = df.drop(index)
                    df.to_csv(DATA_FILE, index=False)
                    st.success("削除しました！")
                    st.rerun()

# --- 4. 分析・統計（グラフ）タブ ---
with tab4:
  st.header("📈 実験データの分析・統計グラフ")
  st.markdown(
      "蓄積データから乳化剤別の傾向や粒径・温度の分布相関を可視化します。"
  )

  if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
    st.write("まだ分析するためのデータがありません。データを登録してください。")
  else:
    df_stats = pd.read_csv(DATA_FILE)

    if len(df_stats) < 1:
      st.write("データが不足しています。")
    else:
      st.subheader("🧪 1. 乳化剤別のデータ登録数・安定傾向")
      if "emulsifier" in df_stats.columns and "stability_text" in df_stats.columns:
        df_stats["status_label"] = df_stats["stability_text"].apply(
            lambda x: "安定" if str(x).startswith("安定") else "不安定"
        )
        counts = (
            df_stats.groupby(["emulsifier", "status_label"])
            .size()
            .unstack(fill_value=0)
        )
        st.bar_chart(counts)
      else:
        st.write("集計に必要なデータ列がありません。")

      st.divider()

      st.subheader(
          "🔬 2. 油分率と試験後の粒子サイズの関係（安定・不安定の分布マップ）"
      )
      if "oil_ratio" in df_stats.columns and "after_d50" in df_stats.columns:
        chart_data = df_stats[
            ["oil_ratio", "after_d50", "status_label", "prescription_no"]
        ]
        st.scatter_chart(
            chart_data,
            x="oil_ratio",
            y="after_d50",
            color="status_label",
            size=50,
        )
      else:
        st.write("散布図を描画するための数値データが不足しています。")
