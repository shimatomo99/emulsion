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
    "🧪 乳化安定性 予測AIアシスタント（写真の追加・削除機能対応版）"
)
st.markdown(
    "<p style='color: #475569; font-size: 1.1em;'>提出時や編集時に顕微鏡写真の追加・差し替え・削除が可能です。</p>",
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
  st.subheader("🌡️ 温度別（5℃ / 40℃）の試験後データ入力タブ")
  st.markdown("それぞれの温度における試験後粒径・濁度・ピーク面積比を入力してください。")

  sub_tab_5c, sub_tab_40c = st.tabs(["❄️ 5℃ 試験データ", "🔥 40℃ 試験データ"])

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

    st.markdown("#### 🔬 5℃ レーザー解析（ピーク面積比）")
    none_ratio_5 = st.checkbox("5℃ ピーク面積比データなし", key="chk_ratio_5")
    if none_ratio_5:
      peak_ratio_5 = 0.0
      st.text("（ピーク面積比: なし）")
    else:
      peak_ratio_5 = st.number_input(
          "5℃ 第1/第2ピーク面積比",
          min_value=0.0,
          value=10.0000,
          step=0.0001,
          format="%.4f",
          key="sub_ratio_5",
      )

    file_5 = st.file_uploader(
        "5℃ 顕微鏡写真ファイルをアップロード",
        type=["png", "jpg", "jpeg"],
        key="file_5",
    )

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

    st.markdown("#### 🔬 40℃ レーザー解析（ピーク面積比）")
    none_ratio_40 = st.checkbox("40℃ ピーク面積比データなし", key="chk_ratio_40")
    if none_ratio_40:
      peak_ratio_40 = 0.0
      st.text("（ピーク面積比: なし）")
    else:
      peak_ratio_40 = st.number_input(
          "40℃ 第1/第2ピーク面積比",
          min_value=0.0,
          value=10.0000,
          step=0.0001,
          format="%.4f",
          key="sub_ratio_40",
      )

    file_40 = st.file_uploader(
        "40℃ 顕微鏡写真ファイルをアップロード",
        type=["png", "jpg", "jpeg"],
        key="file_40",
    )

  st.divider()

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

  if st.button("データ提出 ＆ AI同時予測を実行"):
    folder_map = {
        "アラビアガム": "data/arabic_gum",
        "ポリグリ": "data/polygly",
        "キラヤ": "data/quillaja",
    }
    save_dir = folder_map[emulsifier]
    os.makedirs(save_dir, exist_ok=True)

    image_path_40 = ""
    if file_40 is not None:
      image_path_40 = os.path.join(save_dir, "40c_" + file_40.name)
      with open(image_path_40, "wb") as f:
        f.write(file_40.getvalue())
      file_40.seek(0)

    image_path_5 = ""
    if file_5 is not None:
      image_path_5 = os.path.join(save_dir, "5c_" + file_5.name)
      with open(image_path_5, "wb") as f:
        f.write(file_5.getvalue())
      file_5.seek(0)

    chosen_file = file_40 if file_40 is not None else file_5
    p_count, m_area, m_circ, m_bright, s_bright = (
        extract_advanced_image_features(chosen_file)
        if chosen_file is not None
        else (0.0, 0.0, 0.0, 0.0, 0.0)
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
        "test_days_5": test_days_5,
        "initial_d50": initial_d50,
        "initial_turbidity": initial_turbidity if not none_turb_init else 0.0,
        "after_d50": after_d50_40,
        "delta_d50": delta_d50_40,
        "peak_ratio_40": peak_ratio_40,
        "peak_ratio_5": peak_ratio_5,
        "turbidity_5": turbidity_5,
        "after_d50_5": after_d50_5,
        "delta_d50_5": delta_d50_5,
        "particle_size": after_d50_40,
        "turbidity": turbidity_40,
        "condition": "",
        "stability_text": stability,
        "is_stable": is_stable_val,
        "days": unstable_days,
        "image_path": image_path_40,
        "image_path_5": image_path_5,
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
  pred_peak_ratio = st.number_input(
      "予測時の第1/第2ピーク面積比",
      value=10.0,
      step=0.0001,
      format="%.4f",
      key="pred_peak_ratio",
  )

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
            "peak_ratio_40",
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
            pred_peak_ratio,
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
  st.header("📁 処方グループ別データまとめ・編集 & 乳化剤別CSV出力")
  st.markdown(
      "乳化剤ごとの一括選択や、各データの編集画面からの写真の追加・削除が可能です。"
  )

  if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
    st.write("まだ提出されたデータはありません。")
  else:
    df = pd.read_csv(DATA_FILE)

    st.subheader("📥 乳化剤別・選択式表形式CSVダウンロード")

    st.markdown("#### 🏷️ 乳化剤ごとの一括選択ボタン")
    col_btn_em1, col_btn_em2, col_btn_em3, col_btn_em4 = st.columns(4)

    if "csv_selection_mode" not in st.session_state:
      st.session_state.csv_selection_mode = "全選択"

    if col_btn_em1.button("すべて選択"):
      st.session_state.csv_selection_mode = "全選択"
    if col_btn_em2.button("アラビアガムのみ"):
      st.session_state.csv_selection_mode = "アラビアガム"
    if col_btn_em3.button("ポリグリのみ"):
      st.session_state.csv_selection_mode = "ポリグリ"
    if col_btn_em4.button("キラヤのみ"):
      st.session_state.csv_selection_mode = "キラヤ"

    df["select_label"] = (
        df.index.astype(str)
        + ": "
        + df["prescription_no"].astype(str)
        + " ("
        + df["emulsifier"].astype(str)
        + " / 提出日: "
        + df["date"].astype(str)
        + ")"
    )

    if st.session_state.csv_selection_mode == "全選択":
      default_labels = df["select_label"].tolist()
    else:
      target_emul = st.session_state.csv_selection_mode
      default_labels = df[df["emulsifier"] == target_emul][
          "select_label"
      ].tolist()

    selected_labels = st.multiselect(
        "出力するデータを選択（複数選択・個別変更可）",
        options=df["select_label"].tolist(),
        default=default_labels,
    )

    if selected_labels:
      selected_indices = [
          int(label.split(":")[0]) for label in selected_labels
      ]
      df_export_target = df.loc[selected_indices].copy()
    else:
      df_export_target = df.copy()

    export_columns_map = {
        "prescription_no": "処方コード",
        "emulsifier": "使用した乳化剤",
        "oil_ratio": "油分率 (%)",
        "initial_d50": "初期平均粒径 (nm)",
        "initial_turbidity": "虐待前（初期）濁度",
        "test_days": "40℃試験 期間 (日)",
        "after_d50": "40℃試験後 平均粒径 (nm)",
        "delta_d50": "40℃粒径変動 Δd50 (nm)",
        "turbidity": "40℃ 試験後濁度",
        "peak_ratio_40": "第1/第2ピーク面積比 (40℃)",
        "test_days_5": "5℃試験 期間 (日)",
        "after_d50_5": "5℃試験後 平均粒径 (nm)",
        "delta_d50_5": "5℃粒径変動 Δd50 (nm)",
        "turbidity_5": "5℃ 試験後濁度",
        "peak_ratio_5": "第1/第2ピーク面積比 (5℃)",
        "stability_text": "9か月以内の全体安定性判定",
        "days": "不安定化日数 (日)",
        "particle_count": "画像解析 粒子数",
        "mean_area": "画像解析 平均面積",
        "circularity": "画像解析 円形度",
        "img_mean": "画像解析 輝度平均",
        "img_std": "画像解析 輝度標準偏差",
    }

    existing_cols = [
        c for c in export_columns_map.keys() if c in df_export_target.columns
    ]
    df_formatted_export = df_export_target[existing_cols].rename(
        columns=export_columns_map
    )

    csv_data = df_formatted_export.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 選択したデータをスリム化表形式CSVでダウンロード",
        data=csv_data,
        file_name="emulsion_slim_prescription_table.csv",
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
              col_img1, col_img2, col_txt, col_btn = st.columns([1, 1, 2.2, 0.9])

              with col_img1:
                img_path_5 = str(row.get("image_path_5", ""))
                if os.path.exists(img_path_5):
                  st.image(
                      Image.open(img_path_5),
                      caption=f"❄️ 5℃ ({row.get('date', '')})",
                      width=110,
                  )
                else:
                  st.write("（5℃写真なし）")

              with col_img2:
                img_path_40 = str(row.get("image_path", ""))
                if os.path.exists(img_path_40):
                  st.image(
                      Image.open(img_path_40),
                      caption=f"🔥 40℃ ({row.get('date', '')})",
                      width=110,
                  )
                else:
                  st.write("（40℃写真なし）")

              with col_txt:
                pres_title = str(row.get("prescription_no", ""))
                emuls_name = str(row.get("emulsifier", "不明"))
                st.markdown(f"### 🧪 {pres_title} | {emuls_name}")

                w_amt = row.get("water_amount", "-")
                o_amt = row.get("oil_amount", "-")
                o_rat = row.get("oil_ratio", 0.0)
                d50_after = row.get("after_d50", "-")
                d_d50 = row.get("delta_d50", 0)
                p_ratio_40 = row.get(
                    "peak_ratio_40", row.get("peak1_ratio", "-")
                )

                st.write(
                    f"💧 **油分率**: {float(o_rat):.1f}% （水層 {w_amt}g / 油層"
                    f" {o_amt}g）  \n"
                    f"🔥 **40℃後粒径**: {d50_after}nm (Δ:"
                    f" {float(d_d50):+.1f}nm) / ピーク比:{p_ratio_40}  \n"
                    f"⚖️ **判定**: {row.get('stability_text', '')}"
                )

              with col_btn:

                def get_val(key, default="-"):
                  val = row.get(key, default)
                  return (
                      val
                      if pd.notnull(val) and str(val) != "nan"
                      else default
                  )

                with st.popover("📄 詳細参照"):
                  st.markdown(f"### 🔍 処方データ詳細: {pres_title}")
                  st.write(f"**登録ID**: {index}")
                  st.write(f"**提出日**: {get_val('date')}")
                  st.write(f"**乳化剤**: {emuls_name}")

                  st.markdown("---")
                  st.markdown("#### 💧 処方組成・共通初期値")
                  st.write(f"- 水層の量: {get_val('water_amount')} g")
                  st.write(f"- 油層の量: {get_val('oil_amount')} g")
                  st.write(f"- 油分率: {get_val('oil_ratio')} %")
                  st.write(f"- 初期平均粒径: {get_val('initial_d50')} nm")
                  st.write(f"- 虐待前（初期）濁度: {get_val('initial_turbidity')}")

                  st.markdown("---")
                  st.markdown("#### 🔥 40℃ 試験データ詳細")
                  st.write(f"- 試験期間: {get_val('test_days')} 日")
                  st.write(f"- 試験後平均粒径: {get_val('after_d50')} nm")
                  st.write(f"- 粒径変動 (Δ): {get_val('delta_d50')} nm")
                  st.write(f"- 試験後濁度: {get_val('turbidity')}")
                  st.write(
                      f"- **第1/第2ピーク面積比**: {get_val('peak_ratio_40')}"
                  )

                  st.markdown("---")
                  st.markdown("#### ❄️ 5℃ 試験データ詳細")
                  st.write(f"- 5℃試験後平均粒径: {get_val('after_d50_5')} nm")
                  st.write(f"- 5℃試験後濁度: {get_val('turbidity_5')}")
                  st.write(
                      f"- **第1/第2ピーク面積比**: {get_val('peak_ratio_5')}"
                  )

                  st.markdown("---")
                  st.markdown("#### 📋 総合判定")
                  st.write(f"- 安定性判定: {get_val('stability_text')}")
                  if "不安定" in str(get_val("stability_text")):
                    st.write(f"- 不安定化までの日数: {get_val('days')} 日")

                st.write("")
                with st.popover("🖥️ 編集"):
                  st.markdown(
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
                        value=float(
                            row.get("water_amount", 80.0)
                            if pd.notnull(row.get("water_amount"))
                            else 80.0
                        ),
                        key=f"e_water_{index}",
                    )
                  with col_ew2:
                    edit_oil = st.number_input(
                        "油層の量 (g)",
                        value=float(
                            row.get("oil_amount", 20.0)
                            if pd.notnull(row.get("oil_amount"))
                            else 20.0
                        ),
                        key=f"e_oil_{index}",
                    )

                  st.markdown("#### 🔬 共通初期値の編集（虐待前）")
                  col_ei1, col_ei2 = st.columns(2)
                  with col_ei1:
                    edit_init = st.number_input(
                        "初期平均粒径 (nm)",
                        value=float(
                            row.get("initial_d50", 200.0)
                            if pd.notnull(row.get("initial_d50"))
                            else 200.0
                        ),
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

                  edit_tab_5, edit_tab_40 = st.tabs(
                      ["❄️ 5℃ 試験データ", "🔥 40℃ 試験データ"]
                  )

                  with edit_tab_5:
                    edit_test_days_5 = st.number_input(
                        "5℃試験 期間 (日)",
                        value=int(
                            row.get("test_days_5", 14)
                            if pd.notnull(row.get("test_days_5"))
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
                    edit_ratio_5 = st.number_input(
                        "5℃ 第1/第2ピーク面積比",
                        value=float(
                            row.get("peak_ratio_5", 10.0)
                            if pd.notnull(row.get("peak_ratio_5"))
                            else 10.0
                        ),
                        format="%.4f",
                        step=0.0001,
                        key=f"e_ratio5_{index}",
                    )

                    st.markdown("#### 📸 5℃ 顕微鏡写真の変更・削除")
                    cur_img_5 = str(row.get("image_path_5", ""))
                    if os.path.exists(cur_img_5):
                      st.image(Image.open(cur_img_5), width=100)
                      del_img_5 = st.checkbox(
                          "5℃写真を削除する", key=f"del_img_5_{index}"
                      )
                    else:
                      st.write("（現在5℃写真はありません）")
                      del_img_5 = False

                    new_edit_file_5 = st.file_uploader(
                        "新しい5℃写真をアップロード（差し替え用）",
                        type=["png", "jpg", "jpeg"],
                        key=f"new_file_5_{index}",
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
                        value=float(
                            row.get("after_d50", 220.0)
                            if pd.notnull(row.get("after_d50"))
                            else 220.0
                        ),
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
                    edit_ratio_40 = st.number_input(
                        "40℃ 第1/第2ピーク面積比",
                        value=float(
                            row.get("peak_ratio_40", 10.0)
                            if pd.notnull(row.get("peak_ratio_40"))
                            else 10.0
                        ),
                        format="%.4f",
                        step=0.0001,
                        key=f"e_ratio40_{index}",
                    )

                    st.markdown("#### 📸 40℃ 顕微鏡写真の変更・削除")
                    cur_img_40 = str(row.get("image_path", ""))
                    if os.path.exists(cur_img_40):
                      st.image(Image.open(cur_img_40), width=100)
                      del_img_40 = st.checkbox(
                          "40℃写真を削除する", key=f"del_img_40_{index}"
                      )
                    else:
                      st.write("（現在40℃写真はありません）")
                      del_img_40 = False

                    new_edit_file_40 = st.file_uploader(
                        "新しい40℃写真をアップロード（差し替え用）",
                        type=["png", "jpg", "jpeg"],
                        key=f"new_file_40_{index}",
                    )

                  st.markdown("---")
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

                    folder_map = {
                        "アラビアガム": "data/arabic_gum",
                        "ポリグリ": "data/polygly",
                        "キラヤ": "data/quillaja",
                    }
                    save_dir = folder_map[edit_emulsifier]
                    os.makedirs(save_dir, exist_ok=True)

                    # 写真の削除・更新処理 (5℃)
                    final_img_path_5 = cur_img_5
                    if del_img_5:
                      final_img_path_5 = ""
                    if new_edit_file_5 is not None:
                      final_img_path_5 = os.path.join(
                          save_dir, "5c_" + new_edit_file_5.name
                      )
                      with open(final_img_path_5, "wb") as f:
                        f.write(new_edit_file_5.getvalue())

                    # 写真の削除・更新処理 (40℃)
                    final_img_path_40 = cur_img_40
                    if del_img_40:
                      final_img_path_40 = ""
                    if new_edit_file_40 is not None:
                      final_img_path_40 = os.path.join(
                          save_dir, "40c_" + new_edit_file_40.name
                      )
                      with open(final_img_path_40, "wb") as f:
                        f.write(new_edit_file_40.getvalue())

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
                    df.loc[index, "peak_ratio_40"] = float(edit_ratio_40)
                    df.loc[index, "image_path"] = final_img_path_40

                    # 5℃データの反映
                    df.loc[index, "test_days_5"] = int(edit_test_days_5)
                    df.loc[index, "after_d50_5"] = float(edit_after_5)
                    df.loc[index, "delta_d50_5"] = float(new_delta_5)
                    df.loc[index, "turbidity_5"] = float(edit_turb_5)
                    df.loc[index, "peak_ratio_5"] = float(edit_ratio_5)
                    df.loc[index, "image_path_5"] = final_img_path_5

                    df.loc[index, "stability_text"] = edit_stability
                    df.loc[index, "is_stable"] = int(new_is_stable)

                    df.to_csv(DATA_FILE, index=False)
                    st.success(
                        "すべての変更および写真を正常に保存しました！"
                    )
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
