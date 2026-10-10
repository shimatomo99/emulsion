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

# --- ページ全体のデザイン設定（入力枠を白く見やすく改善） ---
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
    
    /* 📝 すべての入力フォーム・セレクトボックス・数値入力の枠を「白」くくっきりさせる */
    input, select, textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    
    /* Streamlitの各入力ウィジェットの中身（インプットボックス）を強制的に白背景にする */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
        background-color: #ffffff !important;
        border-color: #94a3b8 !important;
    }
    
    /* 数値入力やテキスト入力のラベル文字を見やすく */
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

st.title("🧪 乳化安定性 予測AIアシスタント")
st.markdown(
    "<p style='color: #475569; font-size: 1.1em;'>高精度な画像解析とAI予測で、界面化学・乳化処方の研究をサポートします。</p>",
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
  st.header("実験データの提出")
  st.markdown("新規試作や評価データを登録します。")

  submit_date = st.date_input(
      "実験・提出日", datetime.date.today(), key="sub_date"
  )

  prescription_no = st.text_input(
      "処方番号・派生メモ（例: FORM-001, F200-40℃ など）",
      key="sub_prescription",
  )

  emulsifier = st.selectbox(
      "使用した乳化剤を選択",
      ["アラビアガム", "ポリグリ", "キラヤ"],
      key="sub_emulsifier",
  )

  st.subheader("📊 虐待試験・物性条件の入力")
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
    particle_size = st.number_input(
        "平均粒径 (nm)",
        min_value=0.0,
        max_value=10000.0,
        value=200.0,
        step=10.0,
        key="sub_size",
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
    turbidity = st.number_input(
        "濁度 (吸光度など)",
        min_value=0.0,
        max_value=10.0,
        value=1.0000,
        step=0.0001,
        format="%.4f",
        key="sub_turbidity",
    )

  condition_memo = st.text_input(
      "その他メモ（例: 遠心分離 3000rpm、ホモジナイザー条件など）",
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
          "temperature": temperature,
          "test_days": test_days,
          "particle_size": particle_size,
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
      st.success(
          "データを正常に保存しました！ OpenCVによる粒子解析（検出数:"
          f" {int(p_count)}個、平均円形度: {m_circ:.2f}）も完了しました。"
      )
    else:
      st.warning("写真をアップロードまたは撮影してください。")

# --- 2. 予測・確認タブ ---
with tab2:
  st.header("AIによる安定性予測・高度画像診断")
  st.markdown(
      "画像解析メトリクスと実験条件から、AIが分離リスクと改善アドバイスを提示します。"
  )

  pred_emulsifier = st.selectbox(
      "評価する乳化剤を選択",
      ["アラビアガム", "ポリグリ", "キラヤ"],
      key="pred_emulsifier",
  )

  st.subheader("🔍 予測時に評価する条件の入力")
  col_p1, col_p2 = st.columns(2)
  with col_p1:
    pred_temp = st.number_input(
        "予測時の試験 温度 (℃)", value=60.0, key="pred_temp"
    )
    pred_size = st.number_input(
        "測定された 平均粒径 (nm)", value=200.0, key="pred_size"
    )
  with col_p2:
    pred_test_days = st.number_input(
        "予測時の試験 期間 (日)", value=14, key="pred_test_days"
    )
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

  if st.button("確認ボタン（高度AI予測を実行）"):
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
          feature_cols = [
              "particle_count",
              "mean_area",
              "circularity",
              "img_mean",
              "img_std",
              "temperature",
              "test_days",
              "particle_size",
              "turbidity",
          ]
          for col in feature_cols:
            if col not in train_df.columns:
              train_df[col] = 0.0

          train_clean = train_df.dropna(subset=feature_cols + ["is_stable"])

          if len(train_clean) < 2:
            st.error(
                "十分なパラメータ付き学習データがありません（2件以上必要です"
                "）。"
            )
          else:
            X = train_clean[feature_cols]
            y_class = train_clean["is_stable"]

            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y_class)

            p_count, m_area, m_circ, p_mean, p_std = (
                extract_advanced_image_features(pred_file)
            )
            X_pred = np.array([[
                p_count,
                m_area,
                m_circ,
                p_mean,
                p_std,
                pred_temp,
                pred_test_days,
                pred_size,
                pred_turbidity,
            ]])

            prediction = model.predict(X_pred)[0]
            proba = model.predict_proba(X_pred)[0]
            confidence = max(proba) * 100

            st.subheader("📊 AIの判定結果")

            if prediction == 1:
              st.success(
                  f"判定：**9か月間安定**（予測確信度: {confidence:.1f}%）"
              )
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
                  f"⚠️ **注意喚起**: 過去の同条件データでは平均 **約"
                  f" {avg_unstable_days} 日前後** で分離の兆候があります。"
              )

              st.info("🤖 **AIからの改善ヒント・アドバイス**")
              advice_list = []
              if m_circ < 0.7:
                advice_list.append(
                    "・**粒子の円形度が低下しています**：液滴が変形・合一（コアレッセンス）を起こし始めている可能性があります。界面膜の強度アップを検討してください。"
                )
              if p_count > 150:
                advice_list.append(
                    "・**粒子数が多すぎます（高濃度過密）**：オストワルド熟成や凝集が促進されやすいため、希釈率や乳化剤の再選定が有効です。"
                )
              if pred_size > 300:
                advice_list.append(
                    "・**平均粒径が大きめです**：ホモジナイザーの回転数を上げるか、乳化時間の延長を推奨します。"
                )
              if not advice_list:
                advice_list.append(
                    "・温度条件や試験期間の再確認、および他系統の乳化剤との比較をおすすめします。"
                )

              for advice in advice_list:
                st.write(advice)

            st.subheader("💡 判定の理由と画像解析メトリクス")
            st.write(
                f"- **乳化剤**: {pred_emulsifier} （学習データ数:"
                f" {len(train_clean)}件）"
            )
            st.write(
                f"- **OpenCV解析結果**: 検出粒子数 `{int(p_count)}個` / 平均面積"
                f" `{m_area:.1f} px` / 平均円形度 `{m_circ:.2f}`"
            )
            st.write(
                f"- **入力条件**: 温度 {pred_temp}℃ / 期間"
                f" {pred_test_days}日 / 粒径 {pred_size}nm / 濁度"
                f" {pred_turbidity:.4f}"
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

            turb_val = row.get("turbidity", 0.0)
            turb_str = (
                f"{float(turb_val):.4f}" if pd.notnull(turb_val) else "-"
            )
            p_cnt = row.get("particle_count", 0)
            m_circ = row.get("circularity", 0)

            st.write(
                f"📅 **提出日**: {row.get('date', '不明')}  \n"
                f"🌡️ **試験条件**: 温度 {row.get('temperature', '-')}℃ / 期間"
                f" {row.get('test_days', '-')}日  \n"
                f"🔬 **物性・画像値**: 粒径 {row.get('particle_size', '-')}nm /"
                f" 粒子数 {int(p_cnt)}個 / 円形度"
                f" {float(m_circ):.2f}  \n"
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
              edit_temp = st.number_input(
                  "試験 温度 (℃)",
                  value=float(row.get("temperature", 60.0)),
                  key=f"edit_temp_{index}",
              )
              edit_size = st.number_input(
                  "平均粒径 (nm)",
                  value=float(row.get("particle_size", 200.0)),
                  key=f"edit_size_{index}",
              )
              edit_turb = st.number_input(
                  "濁度",
                  value=float(row.get("turbidity", 1.0)),
                  format="%.4f",
                  step=0.0001,
                  key=f"edit_turb_{index}",
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

                df.loc[index, "prescription_no"] = str(edit_pres)
                df.loc[index, "temperature"] = float(edit_temp)
                df.loc[index, "particle_size"] = float(edit_size)
                df.loc[index, "turbidity"] = float(edit_turb)
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
          "🔬 2. 平均粒径と試験温度の関係（安定・不安定の分布マップ）"
      )
      if (
          "temperature" in df_stats.columns
          and "particle_size" in df_stats.columns
      ):
        chart_data = df_stats[
            [
                "temperature",
                "particle_size",
                "status_label",
                "prescription_no",
            ]
        ]
        st.scatter_chart(
            chart_data,
            x="temperature",
            y="particle_size",
            color="status_label",
            size=50,
        )
      else:
        st.write("散布図を描画するための数値データが不足しています。")
