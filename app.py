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
    "🧪 乳化安定性 予測AIアシスタント（名前順・日付順並び替え対応）"
)
st.markdown(
    "<p style='color: #475569; font-size: 1.1em;'>ベースコードごとにグループ化し、日付順だけでなく名前順（昇順・降順）でも並び替えてデータを管理できます。</p>",
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
      "新しい処方コードを入力し、提出と同時にAIが安定性を診断します。"
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
          "処方コード（例: A200204C, A200204C-40℃ など）",
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

  st.subheader("📊 虐待試験条件の入力")
  col_c1, col_c2 = st.columns(2)
  with col_c1:
    temp_choice = st.radio(
        "虐待試験 温度", ["5℃", "40℃"], horizontal=True, key="sub_temp_choice"
    )
    temperature = 5.0 if temp_choice == "5℃" else 40.0

    none_init = st.checkbox("データなし", key="chk_sub_init")
    if none_init:
      initial_d50 = 0.0
      st.number_input(
          "初期 平均粒径", value=0.0, disabled=True, key="sub_init_dis"
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
          "試験後 平均粒径", value=0.0, disabled=True, key="sub_after_dis"
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

  # --- レーザー解析（粒度分布）セクション ---
  st.subheader("🔬 レーザー解析（粒度分布ピーク範囲）の入力")
  col_l1, col_l2 = st.columns(2)

  with col_l1:
    col_p1_title, col_p1_chk = st.columns([2, 1])
    with col_p1_title:
      st.markdown("#### 山1（第1ピーク）")
    with col_p1_chk:
      none_peak1 = st.checkbox("データなし", key="chk_sub_peak1")

    if none_peak1:
      peak1_min = 0.0
      peak1_max = 0.0
      peak1_ratio = 0.0
      st.text("（第1ピーク: データなし）")
    else:
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

  with col_l2:
    col_p2_title, col_p2_chk = st.columns([2, 1])
    with col_p2_title:
      st.markdown("#### 山2（第2ピーク）")
    with col_p2_chk:
      none_peak2 = st.checkbox("データなし", key="chk_sub_peak2")

    if none_peak2:
      peak2_min = 0.0
      peak2_max = 0.0
      peak2_ratio = 0.0
      st.text("（第2ピーク: データなし）")
    else:
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
      peak2_ratio = st.number_input(
          "第2ピークの面積比 (%)",
          min_value=0.0,
          max_value=100.0,
          value=5.0,
          step=0.5,
          format="%.1f",
          key="sub_p2_ratio",
      )

  delta_d50 = (
      after_d50 - initial_d50
      if not none_after and not none_init
      else 0.0
  )
  st.info(
      f"📈 **変化量プレビュー (ΔD50)**: {delta_d50:+.1f} nm | **第2ピーク比率**:"
      f" {peak2_ratio if not none_peak2 else 0.0}%"
  )

  # --- 濁度測定セクション ---
  st.subheader("💧 濁度測定の入力")
  none_turb = st.checkbox("データなし", key="chk_sub_turb")
  if none_turb:
    turbidity = 0.0000
    st.number_input("濁度", value=0.0, disabled=True, key="sub_turb_dis")
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
      "9か月以内の安定性（教師データ用）",
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

  if st.button("データ提出 ＆ AI同時予測を実行"):
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

      # --- 提出と同時にAI予測を実行 ---
      st.divider()
      st.header("🤖 同時AI予測・診断結果")

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
            "delta_d50": delta_d50,
            "initial_d50": 0.0 if none_init else initial_d50,
            "after_d50": 0.0 if none_after else after_d50,
            "peak1_min": 0.0 if none_peak1 else peak1_min,
            "peak1_max": 0.0 if none_peak1 else peak1_max,
            "peak1_ratio": 0.0 if none_peak1 else peak1_ratio,
            "peak2_min": 0.0 if none_peak2 else peak2_min,
            "peak2_max": 0.0 if none_peak2 else peak2_max,
            "peak2_ratio": 0.0 if none_peak2 else peak2_ratio,
            "temperature": temperature,
            "test_days": test_days,
            "turbidity": 0.0 if none_turb else turbidity,
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
          if not none_peak2 and peak2_ratio > 3.0:
            advice_list.append("・**第2ピーク比率が高めです**。")
          if not none_after and not none_init and delta_d50 > 30:
            advice_list.append("・**粒径変動スピードが大きい**です。")
          if not advice_list:
            advice_list.append("・処方の再検討をおすすめします。")

          for advice in advice_list:
            st.write(advice)

        st.write(
            f"- **乳化剤**: {emulsifier} （学習データ数: {len(train_clean)}件）"
            f" | **使用した有効パラメータ数**: {len(active_feature_cols)}個"
        )
    else:
      st.warning("写真をアップロードまたは撮影してください。")

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

  pred_temp_choice = st.radio(
      "予測時 試験 温度", ["5℃", "40℃"], horizontal=True, key="pred_temp_choice"
  )
  pred_temp = 5.0 if pred_temp_choice == "5℃" else 40.0

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
            pred_temp,
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

    # --- 検索フィルターと並び替え方法（日付順・名前順）の選択 ---
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

      # --- ベースコード抽出 & 同名自動枝番ロジック ---
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

      # --- 並び替え（ソート）の適用 ---
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
                d50_after = row.get("after_d50", "-")
                d_d50 = row.get("delta_d50", 0)
                p2_rat = row.get("peak2_ratio", 5.0)

                st.write(
                    f"📅 **提出日**: {row.get('date', '不明')}  \n"
                    f"💧 **処方組成**: 水層 {w_amt}g / 油層 {o_amt}g （油分率:"
                    f" {float(o_rat):.1f}%）  \n"
                    f"🌡️ **試験条件**: 温度"
                    f" {int(float(row.get('temperature', 40)))}℃ / 期間"
                    f" {row.get('test_days', '-')}日  \n"
                    f"📈 **ダイナミクス**: 初期 {d50_init}nm → 試験後"
                    f" {d50_after}nm (ΔD50: {float(d_d50):+.1f}nm)  \n"
                    f"📊 **第2ピーク比率**: {float(p2_rat)}%  \n"
                    f"⚖️ **判定**: {row.get('stability_text', '')}"
                )
                if "不安定" in str(row.get("stability_text", "")):
                  st.write(
                      f"⏱️ **不安定化までの日数**: {row.get('days', '')} 日"
                  )
                if row.get("condition"):
                  st.write(f"📝 **メモ**: {row.get('condition')}")

              with col3:
                # --- 大画面編集用ポップオーバー ---
                with st.popover("🖥️ 大画面編集"):
                  st.write(
                      f"### ✏️ データの全項目編集 (ID: {index} / 処方:"
                      f" {row.get('prescription_no', '')})"
                  )
                  st.markdown(
                      "すべてのパラメータおよび個別「データなし」設定を変更できます。"
                  )

                  edit_pres = st.text_input(
                      "処方コード（例: A200204C, A200204C-40℃ など）",
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

                  st.markdown("---")
                  st.markdown("#### 💧 水層・油層の量 と データなし設定")
                  col_ew1, col_ew2 = st.columns(2)
                  with col_ew1:
                    e_none_water = st.checkbox(
                        "水層：データなし", key=f"e_none_water_{index}"
                    )
                    if e_none_water:
                      edit_water = 0.0
                      st.text("（水層: 0.0設定）")
                    else:
                      edit_water = st.number_input(
                          "水層の量 (g)",
                          value=float(row.get("water_amount", 80.0)),
                          key=f"e_water_{index}",
                      )
                  with col_ew2:
                    e_none_oil = st.checkbox(
                        "油層：データなし", key=f"e_none_oil_{index}"
                    )
                    if e_none_oil:
                      edit_oil = 0.0
                      st.text("（油層: 0.0設定）")
                    else:
                      edit_oil = st.number_input(
                          "油層の量 (g)",
                          value=float(row.get("oil_amount", 20.0)),
                          key=f"e_oil_{index}",
                      )

                  st.markdown("---")
                  st.markdown("#### 📊 試験条件・平均粒径")
                  col_ec1, col_ec2 = st.columns(2)
                  with col_ec1:
                    cur_temp = float(row.get("temperature", 40.0))
                    edit_temp_choice = st.radio(
                        "試験 温度",
                        ["5℃", "40℃"],
                        index=0 if cur_temp == 5.0 else 1,
                        horizontal=True,
                        key=f"e_temp_choice_{index}",
                    )
                    edit_temp = 5.0 if edit_temp_choice == "5℃" else 40.0

                    e_none_init = st.checkbox(
                        "初期粒径：データなし", key=f"e_none_init_{index}"
                    )
                    if e_none_init:
                      edit_init = 0.0
                      st.text("（初期粒径: 0.0設定）")
                    else:
                      edit_init = st.number_input(
                          "初期平均粒径 (nm)",
                          value=float(row.get("initial_d50", 200.0)),
                          key=f"e_init_{index}",
                      )

                  with col_ec2:
                    edit_test_days = st.number_input(
                        "試験 期間 (日)",
                        value=int(
                            row.get("test_days", 14)
                            if pd.notnull(row.get("test_days"))
                            else 14
                        ),
                        key=f"e_days_{index}",
                    )

                    e_none_after = st.checkbox(
                        "試験後粒径：データなし", key=f"e_none_after_{index}"
                    )
                    if e_none_after:
                      edit_after = 0.0
                      st.text("（試験後粒径: 0.0設定）")
                    else:
                      edit_after = st.number_input(
                          "試験後 平均粒径 (nm)",
                          value=float(row.get("after_d50", 220.0)),
                          key=f"e_after_{index}",
                      )

                  st.markdown("---")
                  st.markdown("#### 🔬 レーザー解析（ピーク範囲）と データなし設定")
                  col_el1, col_el2 = st.columns(2)
                  with col_el1:
                    e_none_p1 = st.checkbox(
                        "山1（第1ピーク）：データなし", key=f"e_none_p1_{index}"
                    )
                    if e_none_p1:
                      edit_p1_min = 0.0
                      edit_p1_max = 0.0
                      edit_p1_ratio = 0.0
                      st.text("（第1ピーク: データなし）")
                    else:
                      edit_p1_min = st.number_input(
                          "第1ピーク 最小値 (nm)",
                          value=float(
                              row.get("peak1_min", 50.0)
                              if pd.notnull(row.get("peak1_min"))
                              else 50.0
                          ),
                          key=f"e_p1min_{index}",
                      )
                      edit_p1_max = st.number_input(
                          "第1ピーク 最大値 (nm)",
                          value=float(
                              row.get("peak1_max", 500.0)
                              if pd.notnull(row.get("peak1_max"))
                              else 500.0
                          ),
                          key=f"e_p1max_{index}",
                      )
                      edit_p1_ratio = st.number_input(
                          "第1ピーク 面積比 (%)",
                          value=float(
                              row.get("peak1_ratio", 95.0)
                              if pd.notnull(row.get("peak1_ratio"))
                              else 95.0
                          ),
                          key=f"e_p1rat_{index}",
                      )

                  with col_el2:
                    e_none_p2 = st.checkbox(
                        "山2（第2ピーク）：データなし", key=f"e_none_p2_{index}"
                    )
                    if e_none_p2:
                      edit_p2_min = 0.0
                      edit_p2_max = 0.0
                      edit_p2_ratio = 0.0
                      st.text("（第2ピーク: データなし）")
                    else:
                      edit_p2_min = st.number_input(
                          "第2ピーク 最小値 (nm)",
                          value=float(
                              row.get("peak2_min", 500.0)
                              if pd.notnull(row.get("peak2_min"))
                              else 500.0
                          ),
                          key=f"e_p2min_{index}",
                      )
                      edit_p2_max = st.number_input(
                          "第2ピーク 最大値 (nm)",
                          value=float(
                              row.get("peak2_max", 5000.0)
                              if pd.notnull(row.get("peak2_max"))
                              else 5000.0
                          ),
                          key=f"e_p2max_{index}",
                      )
                      edit_p2_ratio = st.number_input(
                          "第2ピーク 面積比 (%)",
                          value=float(
                              row.get("peak2_ratio", 5.0)
                              if pd.notnull(row.get("peak2_ratio"))
                              else 5.0
                          ),
                          key=f"e_p2rat_{index}",
                      )

                  st.markdown("---")
                  st.markdown("#### 💧 濁度・メモ・安定性")
                  e_none_turb = st.checkbox(
                      "濁度：データなし", key=f"e_none_turb_{index}"
                  )
                  if e_none_turb:
                    edit_turbidity = 0.0
                    st.text("（濁度: 0.0設定）")
                  else:
                    edit_turbidity = st.number_input(
                        "濁度",
                        value=float(
                            row.get("turbidity", 1.0)
                            if pd.notnull(row.get("turbidity"))
                            else 1.0
                        ),
                        format="%.4f",
                        key=f"e_turb_{index}",
                    )

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
                    new_delta = (
                        edit_after - edit_init
                        if not e_none_after and not e_none_init
                        else 0.0
                    )
                    new_is_stable = (
                        1 if edit_stability.startswith("安定") else 0
                    )

                    df.loc[index, "prescription_no"] = str(edit_pres)
                    df.loc[index, "emulsifier"] = edit_emulsifier
                    df.loc[index, "water_amount"] = float(edit_water)
                    df.loc[index, "oil_amount"] = float(edit_oil)
                    df.loc[index, "oil_ratio"] = float(new_o_rat)
                    df.loc[index, "temperature"] = float(edit_temp)
                    df.loc[index, "test_days"] = int(edit_test_days)
                    df.loc[index, "initial_d50"] = float(edit_init)
                    df.loc[index, "after_d50"] = float(edit_after)
                    df.loc[index, "delta_d50"] = float(new_delta)
                    df.loc[index, "peak1_min"] = float(edit_p1_min)
                    df.loc[index, "peak1_max"] = float(edit_p1_max)
                    df.loc[index, "peak1_ratio"] = float(edit_p1_ratio)
                    df.loc[index, "peak2_min"] = float(edit_p2_min)
                    df.loc[index, "peak2_max"] = float(edit_p2_max)
                    df.loc[index, "peak2_ratio"] = float(edit_p2_ratio)
                    df.loc[index, "turbidity"] = float(edit_turbidity)
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
