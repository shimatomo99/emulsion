import datetime
import os
import numpy as np
import pandas as pd
from PIL import Image
import sklearn
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# データの保存先CSV
DATA_FILE = "dataset.csv"

st.title("🧪 乳化安定性 予測AIアプリ")

# タブの切り替え
tab1, tab2, tab3, tab4 = st.tabs([
    "📁 データ提出モード",
    "🔍 予測・確認モード",
    "📊 データ履歴・見返し",
    "📈 分析・統計（グラフ）",
])


# --- 画像から簡易的な特徴量を取り出す関数 ---
def extract_image_features(image_file):
  try:
    img = Image.open(image_file).convert("L")  # グレースケール変換
    img = img.resize((100, 100))  # サイズ統一
    arr = np.array(img)
    mean_val = np.mean(arr)  # 平均明るさ
    std_val = np.std(arr)  # 濃淡のバラつき
    return mean_val, std_val
  except Exception:
    return 0.0, 0.0


# --- 1. データ提出タブ ---
with tab1:
  st.header("実験データの提出")

  submit_date = st.date_input(
      "実験・提出日", datetime.date.today(), key="sub_date"
  )

  prescription_no = st.text_input(
      "処方番号（例: FORM-001 など）", key="sub_prescription"
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
      "その他メモ（例: 遠心分離 3000rpm など）", key="sub_condition"
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
      "コロイドの顕微鏡写真をアップロード (ドラッグ＆ドロップ可)",
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
      img = Image.open(uploaded_file)
      img.save(image_path)

      mean_v, std_v = extract_image_features(uploaded_file)
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
          "img_mean": mean_v,
          "img_std": std_v,
      }])

      if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
        try:
          df = pd.read_csv(DATA_FILE)
          df = pd.concat([df, new_data], ignore_index=True)
        except Exception:
          df = new_data
      else:
        df = new_data

      # 保存前に文字列列の型を文字列に強制変換してエラーを防ぐ
      df["prescription_no"] = df["prescription_no"].astype(str)
      df["condition"] = df["condition"].astype(str)
      df.to_csv(DATA_FILE, index=False)
      st.success("データを正常に保存しました！")
    else:
      st.warning("写真をアップロードしてください。")

# --- 2. 予測・確認タブ ---
with tab2:
  st.header("AIによる安定性予測・理由付け")
  st.info(
      "画像特徴量に加え、温度・期間・粒径・濁度などの実験条件を総合的に考慮してAIが予測します。"
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
      "確認用コロイドの顕微鏡写真をアップロード",
      type=["png", "jpg", "jpeg"],
      key="pred_file",
  )

  if st.button("確認ボタン（AI予測を実行）"):
    if pred_file is not None:
      if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        st.warning("学習データがまだありません。データを登録してください。")
      else:
        df = pd.read_csv(DATA_FILE)
        train_df = df[df["emulsifier"] == pred_emulsifier]

        if len(train_df) < 2:
          train_df = df  # データが少なければ全体を使う

        if len(train_df) < 2:
          st.error(
              "AIが学習するためのデータが最低2件以上必要です。データを追加し"
              "てください。"
          )
        else:
          feature_cols = [
              "img_mean",
              "img_std",
              "temperature",
              "test_days",
              "particle_size",
              "turbidity",
          ]
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

            p_mean, p_std = extract_image_features(pred_file)
            X_pred = np.array([[
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
            else:
              st.warning(
                  f"判定：**9か月以内に不安定化の恐れあり**（予測確信度:"
                  f" {confidence:.1f}%）"
              )

            st.subheader("💡 判定の理由と分析")
            st.write(
                f"- **乳化剤**: {pred_emulsifier} （学習データ数:"
                f" {len(train_clean)}件）"
            )
            st.write(
                f"- **入力条件**: 温度 {pred_temp}℃ / 期間"
                f" {pred_test_days}日 / 粒径 {pred_size}nm / 濁度"
                f" {pred_turbidity:.4f}"
            )
            st.write(
                f"- **画像特徴量**: 平均明るさ `{p_mean:.1f}`、粒子の濃淡バラつき"
                f" `{p_std:.1f}`"
            )

            if p_std > train_clean["img_std"].mean():
              st.write(
                  "  - ⚠️ 過去のデータと比較して**粒子のバラつき（標準偏差）が"
                  "大きめ**です。これが不安定化のリスク要因となっている可能性"
                  "があります。"
              )
            else:
              st.write(
                  "  - ✨ 過去の安定データと近い**良好な分散状態**を示していま"
                  "す。"
              )
    else:
      st.warning("確認用の写真をアップロードしてください。")

# --- 3. データ履歴・見返しタブ（編集機能付き） ---
with tab3:
  st.header("📁 提出データ履歴・見返し")
  st.info(
      "これまで登録した実験データを確認・編集・削除できます。（最新20件まで表示）"
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

    search_query = st.text_input(
        "🔍 処方番号で検索（部分一致）", "", key="hist_search_pres"
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
                f"### 🏷️ 処方番号: {pres_no} （乳化剤: {row.get('emulsifier', '')}）"
            )

            turb_val = row.get("turbidity", 0.0)
            turb_str = (
                f"{float(turb_val):.4f}" if pd.notnull(turb_val) else "-"
            )

            st.write(
                f"📅 **提出日**: {row.get('date', '不明')}  \n"
                f"🌡️ **試験条件**: 温度 {row.get('temperature', '-')}℃ / 期間"
                f" {row.get('test_days', '-')}日  \n"
                f"🔬 **物性値**: 平均粒径 {row.get('particle_size', '-')}nm /"
                f" 濁度 {turb_str}  \n"
                f"⚖️ **判定**: {row.get('stability_text', '')}"
            )
            if "不安定" in str(row.get("stability_text", "")):
              st.write(f"⏱️ **不安定化までの日数**: {row.get('days', '')} 日")

          with col3:
            with st.popover("✏️ 編集"):
              st.write("### データの編集")
              edit_pres = st.text_input(
                  "処方番号",
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
                # データを更新する前に、確実に各列を正しい型に合わせる
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
  st.info(
      "蓄積された実験データから、乳化剤ごとの傾向や、粒径・温度と安定性の関係を視覚的に確認できます。"
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