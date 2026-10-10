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
    "🧪 乳化安定性 予測AIアシスタント（未測定項目自動除外AI予測対応）"
)
st.markdown(
    "<p style='color: #475569; font-size: 1.1em;'>データなし項目（0）をAI予測時から自動除外し、測定済みデータのみで高精度に推察します。</p>",
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
    "📁 データ提出モード",
    "🔍 予測・確認モード",
    "📊 データ履歴・見返し",
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


# --- 1. データ提出タブ ---
with tab1:
  st.header("実験データの提出（個別データなしボタン対応）")
  st.markdown(
      "各項目の横にある「データなし」にチェックを入れると、値が自動的に"
      "「0」または「データなし」になります。"
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
          "処方番号・派生メモ",
          value="データなし",
          disabled=True,
          key="sub_prescription_dis",
      )
    else:
      prescription_no = st.text_input(
          "処方番号・派生メモ（例: FORM-001, F200-40℃ など）",
          value="FORM-001",
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
          "水層の量 (g または mL)",
          value=0.0,
          disabled=True,
          key="sub_water_dis",
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
          "油層の量 (g または mL)",
          value=0.0,
          disabled=True,
          key="sub_oil_dis",
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

  st.subheader("📊 虐待試験・粒度分布ピーク範囲の入力")
  col_c1, col_c2 = st.columns(2)
  with col_c1:
    temperature = st.number_input(
        "虐待試験 温度 (℃)",
        min_value=0.0,
        max_value=120.0,
        value=60.0,
        step=1.0,
        key="sub_temp",
    )

    none_init = st.checkbox("データなし", key="chk_sub_init")
    if none_init:
      initial_d50 = 0.0
      st.number_input(
          "製造直後の初期 平均粒径 (nm)",
          value=0.0,
          disabled=True,
          key="sub_init_dis",
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

    st.markdown("---")
    st.markdown("#### 山1（第1ピーク）の範囲と面積比")
    peak1_min = st.number_input(
        "第1ピーク 最小値 (nm)",
        min_value=0.0,
        value=50.0,
        step=10.0,
        key="sub_p1_min",
    )
    peak1_max = st.number_input(
        "第1ピーク 最大値 (nm)",
        min_value=0.0,
        value=500.0,
        step=10.0,
        key="sub_p1_max",
    )
    peak1_ratio = st.number_input(
        "第1ピークの面積比 (%)",
        min_value=0.0,
        max_value=100.0,
        value=95.0,
        step=0.5,
        format="%.1f",
        key="sub_p1_ratio",
    )

    none_coarse = st.checkbox("データなし", key="chk_sub_coarse")
    if none_coarse:
      coarse_ratio = 0.0
      st.number_input(
          "粗大粒子割合（1 μm以上 %）",
          value=0.0,
          disabled=True,
          key="sub_coarse_dis",
      )
    else:
      coarse_ratio = st.number_input(
          "粗大粒子割合（1 μm以上の積分比 %）",
          min_value=0.0,
          max_value=100.0,
          value=1.5,
          step=0.1,
          format="%.1f",
          key="sub_coarse",
      )

  with col_c2:
    test_days = st.number_input(
        "虐待試験 期間 (日)",
        min_value=1,
        max_value=365,
        value=14,
        step=1,
        key="sub_test_days",
    )

    none_after = st.checkbox("データなし", key="chk_sub_after")
    if none_after:
      after_d50 = 0.0
      st.number_input(
          "試験終了後の 平均粒径 (nm)",
          value=0.0,
          disabled=True,
          key="sub_after_dis",
      )
    else:
      after_d50 = st.number_input(
          "試験終了後の 平均粒径 (nm)",
          min_value=0.0,
          max_value=10000.0,
          value=220.0,
          step=10.0,
          key="sub_after_d50",
      )

    st.markdown("---")
    st.markdown("#### 山2（第2ピーク / 粗大・合一）の範囲と面積比")
    peak2_min = st.number_input(
        "第2ピーク 最小値 (nm)",
        min_value=0.0,
        value=500.0,
        step=10.0,
        key="sub_p2_min",
    )
    peak2_max = st.number_input(
        "第2ピーク 最大値 (nm)",
        min_value=0.0,
        value=5000.0,
        step=10.0,
        key="sub_p2_max",
    )

    none_p2_rat = st.checkbox("データなし", key="chk_sub_p2_rat")
    if none_p2_rat:
      peak2_ratio = 0.0
      st.number_input(
          "第2ピークの面積比 (%)",
          value=0.0,
          disabled=True,
          key="sub_p2_rat_dis",
      )
    else:
      peak2_ratio = st.number_input(
          "第2ピークの面積比 (%)",
          min_value=0.0,
          max_value=100.0,
          value=5.0,
          step=0.5,
          format="%.1f",
          key="sub_p2_ratio",
      )

  delta_d50 = after_d50 - initial_d50
  st.info(
      f"📈 **変化量プレビュー (ΔD50)**: {delta_d50:+.1f} nm | **第2ピーク比率**:"
      f" {peak2_ratio}%"
  )

  none_turb = st.checkbox("データなし", key="chk_sub_turb")
  if none_turb:
    turbidity = 0.0000
    st.number_input(
        "濁度 (吸光度など)", value=0.0, disabled=True, key="sub_turb_dis"
    )
  else:
    turbidity = st.number_input(
        "濁度 (吸光度など)",
        min_value=0.0,
        max_value=10.0,
        value=1.0000,
        step=0.0001,
        format="%.4f",
        key="sub_turbidity",
    )

  none_cond = st.checkbox("データなし", key="chk_sub_cond")
  if none_cond:
    condition_memo = "データなし"
    st.text_input(
        "その他メモ", value="データなし", disabled=True, key="sub_cond_dis"
    )
  else:
    condition_memo = st.text_input(
        "その他メモ（例: 遠心分離 3000rpm、ホモジナイザー条件など）",
        value="",
        key="sub_condition",
    )

  stability = st.radio(
      "9か月以内の安定性",
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

  uploaded_file = st.file_uploader(
      "📸 コロイドの顕微鏡写真をアップロード（スマホからはその場でカメラ撮影も可）",
      type=["png", "jpg", "jpeg"],
      key="sub_file",
  )

  if st.button("データ提出"):
    if uploaded_file is not None:
      folder_map = {
          "アラビアガム": "data/arabic_gum",
          "ポリグリ": "data/polygly",
          "キラヤ": "data/quillaja",
      }
      save_dir = folder_map[emulsifier]
      os.makedirs(save_dir, exist_ok=True)

      image_path = os.path.join(save_dir, uploaded_file.name)

      img_bytes_for_save = uploaded_file.getvalue()
      with open(image_path, "wb") as f:
        f.write(img_bytes_for_save)

      uploaded_file.seek(0)
      p_count, m_area, m_circ, m_bright, s_bright = (
          extract_advanced_image_features(uploaded_file)
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
          "temperature": temperature,
          "test_days": test_days,
          "initial_d50": initial_d50,
          "after_d50": after_d50,
          "delta_d50": delta_d50,
          "coarse_ratio": coarse_ratio,
          "peak1_min": peak1_min,
          "peak1_max": peak1_max,
          "peak1_ratio": peak1_ratio,
          "peak2_min": peak2_min,
          "peak2_max": peak2_max,
          "peak2_ratio": peak2_ratio,
          "particle_size": after_d50,
          "turbidity": turbidity,
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
      st.success("データを正常に保存しました！")
    else:
      st.warning("写真をアップロードまたは撮影してください。")

# --- 2. 予測・確認タブ ---
with tab2:
  st.header("AIによる安定性予測・ダイナミクス診断")
  st.markdown("未測定項目は「データなし」にチェックを入れて予測できます。")

  pred_emulsifier = st.selectbox(
      "評価する乳化剤を選択",
      ["アラビアガム", "ポリグリ", "キラヤ"],
      key="pred_emulsifier",
  )

  st.subheader("💧 予測時の処方組成（水層・油層の量）の入力")
  col_pw1, col_pw2 = st.columns(2)
  with col_pw1:
    none_p_water = st.checkbox("データなし", key="chk_pred_water")
    if none_p_water:
      pred_water = 0.0
      st.number_input(
          "予測時の水層の量 (g)", value=0.0, disabled=True, key="pred_water_dis"
      )
    else:
      pred_water = st.number_input(
          "予測時の水層の量 (g または mL)",
          min_value=0.0,
          max_value=10000.0,
          value=80.0,
          step=1.0,
          key="pred_water",
      )
  with col_pw2:
    none_p_oil = st.checkbox("データなし", key="chk_pred_oil")
    if none_p_oil:
      pred_oil = 0.0
      st.number_input(
          "予測時の油層の量 (g)", value=0.0, disabled=True, key="pred_oil_dis"
      )
    else:
      pred_oil = st.number_input(
          "予測時の油層の量 (g または mL)",
          min_value=0.0,
          max_value=10000.0,
          value=20.0,
          step=1.0,
          key="pred_oil",
      )

  pred_total = pred_water + pred_oil
  pred_oil_ratio = (
      (pred_oil / pred_total * 100.0) if pred_total > 0 else 0.0
  )
  st.info(
      f"📊 **予測時 油分率**: {pred_oil_ratio:.1f}% （総量: {pred_total:.1f}g）"
  )

  st.subheader("🔍 予測時に評価する条件の入力")
  col_p1, col_p2 = st.columns(2)
  with col_p1:
    pred_temp = st.number_input(
        "予測時の試験 温度 (℃)", value=60.0, key="pred_temp"
    )

    none_p_init = st.checkbox("データなし", key="chk_pred_init")
    if none_p_init:
      pred_init_d50 = 0.0
      st.number_input(
          "予測時の初期 平均粒径 (nm)",
          value=0.0,
          disabled=True,
          key="pred_init_dis",
      )
    else:
      pred_init_d50 = st.number_input(
          "予測時の初期 平均粒径 (nm)",
          min_value=0.0,
          max_value=10000.0,
          value=200.0,
          step=10.0,
          key="pred_init_d50",
      )

    st.markdown("---")
    st.markdown("#### 予測時 山1（第1ピーク）の範囲と面積比")
    pred_p1_min = st.number_input(
        "第1ピーク 最小値 (nm)",
        min_value=0.0,
        value=50.0,
        step=10.0,
        key="pred_p1_min",
    )
    pred_p1_max = st.number_input(
        "第1ピーク 最大値 (nm)",
        min_value=0.0,
        value=500.0,
        step=10.0,
        key="pred_p1_max",
    )
    pred_p1_ratio = st.number_input(
        "第1ピークの面積比 (%)",
        min_value=0.0,
        max_value=100.0,
        value=95.0,
        step=0.5,
        format="%.1f",
        key="pred_p1_ratio",
    )

    none_p_coarse = st.checkbox("データなし", key="chk_pred_coarse")
    if none_p_coarse:
      pred_coarse = 0.0
      st.number_input(
          "予測時の粗大粒子割合",
          value=0.0,
          disabled=True,
          key="pred_coarse_dis",
      )
    else:
      pred_coarse = st.number_input(
          "予測時の粗大粒子割合（1 μm以上 %）",
          min_value=0.0,
          max_value=100.0,
          value=1.5,
          step=0.1,
          format="%.1f",
          key="pred_coarse",
      )

  with col_p2:
    pred_test_days = st.number_input(
        "予測時の試験 期間 (日)", value=14, key="pred_test_days"
    )

    none_p_after = st.checkbox("データなし", key="chk_pred_after")
    if none_p_after:
      pred_after_d50 = 0.0
      st.number_input(
          "予測時の試験後 平均粒径 (nm)",
          value=0.0,
          disabled=True,
          key="pred_after_dis",
      )
    else:
      pred_after_d50 = st.number_input(
          "予測時の試験後 平均粒径 (nm)",
          min_value=0.0,
          max_value=10000.0,
          value=220.0,
          step=10.0,
          key="pred_after_d50",
      )

    st.markdown("---")
    st.markdown("#### 予測時 山2（第2ピーク）の範囲と面積比")
    pred_p2_min = st.number_input(
        "第2ピーク 最小値 (nm)",
        min_value=0.0,
        value=500.0,
        step=10.0,
        key="pred_p2_min",
    )
    pred_p2_max = st.number_input(
        "第2ピーク 最大値 (nm)",
        min_value=0.0,
        value=5000.0,
        step=10.0,
        key="pred_p2_max",
    )

    none_p_p2_rat = st.checkbox("データなし", key="chk_pred_p2_rat")
    if none_p_p2_rat:
      pred_p2_ratio = 0.0
      st.number_input(
          "第2ピーク面積比 (%)",
          value=0.0,
          disabled=True,
          key="pred_p2_rat_dis",
      )
    else:
      pred_p2_ratio = st.number_input(
          "第2ピークの面積比 (%)",
          min_value=0.0,
          max_value=100.0,
          value=5.0,
          step=0.5,
          format="%.1f",
          key="pred_p2_ratio",
      )

  pred_delta_d50 = (
      pred_after_d50 - pred_init_d50
      if not none_p_after and not none_p_init
      else 0.0
  )
  st.info(
      f"📈 **予測時 ΔD50**: {pred_delta_d50:+.1f} nm | **油分率**: "
      f"{pred_oil_ratio:.1f}% | **第2ピーク比率**: {pred_p2_ratio}%"
  )

  none_p_turb = st.checkbox("データなし", key="chk_pred_turb")
  if none_p_turb:
    pred_turbidity = 0.0000
    st.number_input("濁度", value=0.0, disabled=True, key="pred_turb_dis")
  else:
    pred_turbidity = st.number_input(
        "測定された 濁度",
        value=1.0000,
        step=0.0001,
        format="%.4f",
        key="pred_turbidity",
    )

  pred_file = st.file_uploader(
      "📸 確認用コロイドの顕微鏡写真をアップロード（またはスマホカメラで撮影）",
      type=["png", "jpg", "jpeg"],
      key="pred_file",
  )

  if st.button("確認ボタン（高精度AI予測を実行）"):
    if pred_file is not None:
      if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        st.warning("学習データがまだありません。データを登録してください。")
      else:
        df = pd.read_csv(DATA_FILE)
        train_df = df[df["emulsifier"] == pred_emulsifier]

        if len(train_df) < 2:
          train_df = df

        if len(train_df) < 2:
          st.error(
              "AIが学習するためのデータが最低2件以上必要です。データを追加し"
              "てください。"
          )
        else:
          # --- ★「データなし（0）」の項目をAIの学習・予測から自動除外する仕組み ---
          base_feature_cols = [
              "oil_ratio",
              "water_amount",
              "oil_amount",
              "delta_d50",
              "initial_d50",
              "after_d50",
              "coarse_ratio",
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

          # 予測時にユーザーが「データなし（0）」にした項目は特徴量から除外する
          active_feature_cols = []
          pred_values = []

          # 各入力とフラグの対応チェック
          val_map = {
              "oil_ratio": pred_oil_ratio,
              "water_amount": 0.0 if none_p_water else pred_water,
              "oil_amount": 0.0 if none_p_oil else pred_oil,
              "delta_d50": pred_delta_d50,
              "initial_d50": 0.0 if none_p_init else pred_init_d50,
              "after_d50": 0.0 if none_p_after else pred_after_d50,
              "coarse_ratio": 0.0 if none_p_coarse else pred_coarse,
              "peak1_min": pred_p1_min,
              "peak1_max": pred_p1_max,
              "peak1_ratio": pred_p1_ratio,
              "peak2_min": pred_p2_min,
              "peak2_max": pred_p2_max,
              "peak2_ratio": 0.0 if none_p_p2_rat else pred_p2_ratio,
              "temperature": pred_temp,
              "test_days": pred_test_days,
              "turbidity": 0.0 if none_p_turb else pred_turbidity,
          }

          p_count, m_area, m_circ, p_mean, p_std = (
              extract_advanced_image_features(pred_file)
          )
          img_map = {
              "particle_count": p_count,
              "mean_area": m_area,
              "circularity": m_circ,
              "img_mean": p_mean,
              "img_std": p_std,
          }
          val_map.update(img_map)

          # 0ではない（＝データがある）項目だけをAIの学習・予測に使用する
          for col in base_feature_cols:
            if col in train_df.columns:
              # 履歴データ側で0でない、かつ予測入力側でも有効な数値の項目を採用
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

          train_clean = train_df.dropna(
              subset=active_feature_cols + ["is_stable"]
          )

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

          st.subheader("📊 AIの判定結果")

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
            if not none_p_oil and pred_oil_ratio > 40.0:
              advice_list.append(
                  "・**油分率が高めの処方です**：乳化剤の増量を推奨します。"
              )
            if not none_p_p2_rat and pred_p2_ratio > 3.0:
              advice_list.append("・**第2ピーク比率が高めです**。")
            if not none_p_after and not none_p_init and pred_delta_d50 > 30:
              advice_list.append("・**粒径変動スピードが大きい**です。")
            if not advice_list:
              advice_list.append("・処方の再検討をおすすめします。")

            for advice in advice_list:
              st.write(advice)

          st.subheader("💡 判定の理由と使用された有効指標")
          st.write(
              f"- **乳化剤**: {pred_emulsifier} （学習データ数:"
              f" {len(train_clean)}件）"
          )
          st.write(
              f"- **AIが使用した有効パラメータ数**: {len(active_feature_cols)}個"
              "（未測定項目は除外済み）"
          )
    else:
      st.warning("確認用の写真をアップロードまたは撮影してください。")

# --- 3. データ履歴・見返しタブ ---
with tab3:
  st.header("📁 提出データ履歴・見返し")
  st.markdown("過去の実験データの検索、詳細確認、編集・削除が可能です。")

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

    search_query = st.text_input(
        "🔍 処方番号・温度タグで検索（例: F200、40℃、FORM-001 など）",
        "",
        key="hist_search_pres",
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
      filter_emulsifier = st.selectbox(
          "乳化剤で絞り込み",
          ["すべて", "アラビアガム", "ポリグリ", "キラヤ"],
          key="hist_filter_emulsifier",
      )
    with col_f2:
      filter_stability = st.selectbox(
          "安定性で絞り込み",
          ["すべて", "安定", "不安定"],
          key="hist_filter_stability",
      )

    df_display = df.copy()

    if search_query:
      df_display["prescription_no"] = (
          df_display["prescription_no"].astype(str).fillna("")
      )
      df_display = df_display[
          df_display["prescription_no"].str.contains(search_query, case=False)
      ]

    if filter_emulsifier != "すべて":
      df_display = df_display[df_display["emulsifier"] == filter_emulsifier]

    if filter_stability == "安定":
      df_display = df_display[
          df_display["stability_text"].str.startswith("安定", na=False)
      ]
    elif filter_stability == "不安定":
      df_display = df_display[
          df_display["stability_text"].str.startswith("不安定", na=False)
      ]

    if len(df_display) == 0:
      st.write("該当するデータはありません。")
    else:
      if "date" in df_display.columns:
        df_display = df_display.sort_values(by="date", ascending=False)

      total_count = len(df_display)
      df_display = df_display.head(20)

      st.write(
          f"該当件数: **{total_count}件** （うち、最新の **{len(df_display)}件**"
          " を表示中）"
      )

      for index, row in df_display.iterrows():
        with st.container(border=True):
          col1, col2, col3 = st.columns([1.2, 2.8, 0.8])

          with col1:
            img_path = str(row.get("image_path", ""))
            if os.path.exists(img_path):
              image = Image.open(img_path)
              st.image(
                  image,
                  caption=f"撮影日: {row.get('date', '不明')}",
                  width=140,
              )
            else:
              st.write("（写真なし）")

          with col2:
            pres_no = row.get("prescription_no", "未設定")
            st.markdown(
                f"### 🏷️ 処方番号・タグ: {pres_no} （乳化剤:"
                f" {row.get('emulsifier', '')}）"
            )

            w_amt = row.get("water_amount", "-")
            o_amt = row.get("oil_amount", "-")
            o_rat = row.get("oil_ratio", 0.0)
            d50_init = row.get("initial_d50", "-")
            d50_after = row.get("after_d50", "-")
            d_d50 = row.get("delta_d50", 0)
            p2_rat = row.get("peak2_ratio", 5.0)

            st.write(
                f"📅 **提出日**: {row.get('date', '不明')}  \n"
                f"💧 **処方組成**: 水層 {w_amt}g / 油層 {o_amt}g （油分率:"
                f" {float(o_rat):.1f}%）  \n"
                f"🌡️ **試験条件**: 温度 {row.get('temperature', '-')}℃ / 期間"
                f" {row.get('test_days', '-')}日  \n"
                f"📈 **ダイナミクス**: 初期 {d50_init}nm → 試験後"
                f" {d50_after}nm (ΔD50: {float(d_d50):+.1f}nm)  \n"
                f"📊 **第2ピーク比率**: {float(p2_rat)}%  \n"
                f"⚖️ **判定**: {row.get('stability_text', '')}"
            )
            if "不安定" in str(row.get("stability_text", "")):
              st.write(f"⏱️ **不安定化までの日数**: {row.get('days', '')} 日")
            if row.get("condition"):
              st.write(f"📝 **メモ**: {row.get('condition')}")

          with col3:
            with st.popover("✏️ 編集"):
              st.write("### データの編集")
              edit_pres = st.text_input(
                  "処方番号・タグ",
                  value=str(row.get("prescription_no", "")),
                  key=f"edit_pres_{index}",
              )
              edit_water = st.number_input(
                  "水層の量 (g)",
                  value=float(row.get("water_amount", 80.0)),
                  key=f"edit_water_{index}",
              )
              edit_oil = st.number_input(
                  "油層の量 (g)",
                  value=float(row.get("oil_amount", 20.0)),
                  key=f"edit_oil_{index}",
              )
              edit_temp = st.number_input(
                  "試験 温度 (℃)",
                  value=float(row.get("temperature", 60.0)),
                  key=f"edit_temp_{index}",
              )
              edit_init = st.number_input(
                  "初期平均粒径 (nm)",
                  value=float(row.get("initial_d50", 200.0)),
                  key=f"edit_init_{index}",
              )
              edit_after = st.number_input(
                  "試験後 平均粒径 (nm)",
                  value=float(row.get("after_d50", 220.0)),
                  key=f"edit_after_{index}",
              )
              edit_cond = st.text_input(
                  "その他メモ",
                  value=(
                      str(row.get("condition", ""))
                      if pd.notnull(row.get("condition"))
                      else ""
                  ),
                  key=f"edit_cond_{index}",
              )

              if st.button("💾 変更を保存", key=f"save_edit_{index}"):
                df["prescription_no"] = df["prescription_no"].astype(str)
                df["condition"] = df["condition"].astype(str)

                tot = edit_water + edit_oil
                new_o_rat = (edit_oil / tot * 100.0) if tot > 0 else 0.0

                df.loc[index, "prescription_no"] = str(edit_pres)
                df.loc[index, "water_amount"] = float(edit_water)
                df.loc[index, "oil_amount"] = float(edit_oil)
                df.loc[index, "oil_ratio"] = float(new_o_rat)
                df.loc[index, "temperature"] = float(edit_temp)
                df.loc[index, "initial_d50"] = float(edit_init)
                df.loc[index, "after_d50"] = float(edit_after)
                df.loc[index, "delta_d50"] = float(edit_after - edit_init)
                df.loc[index, "particle_size"] = float(edit_after)
                df.loc[index, "condition"] = str(edit_cond)
                df.to_csv(DATA_FILE, index=False)
                st.success("更新しました！")
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
