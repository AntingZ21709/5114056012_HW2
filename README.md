# 紅酒品質預測專案

本專案使用機器學習方法，基於紅酒的物理化學特性預測其品質評分。這是一個使用 Python 和 Streamlit 建立的互動式資料分析應用程式。

## 專案結構

```
├── app.py                 # 主要的 Streamlit 應用程式
├── CRISP-DM.txt          # CRISP-DM 六大步驟完整報告
├── NotebookLM.txt        # 多元線性回歸分析研究摘要
└── README.md             # 專案說明文件（本文件）
```

## 功能特色

- 完整的資料預處理流程
  - 缺失值處理
  - 離群值偵測
  - 特徵標準化
  - 訓練/測試資料切分

- 多元線性回歸模型
  - 特徵選擇（SelectKBest）
  - 模型訓練與評估
  - 視覺化分析

- 互動式介面
  - 使用 Streamlit 建立的網頁應用程式
  - 動態特徵選擇
  - 即時模型更新
  - 視覺化結果展示

## 安裝需求

確保您已安裝以下 Python 套件：

```bash
pip install streamlit pandas numpy seaborn matplotlib scikit-learn
```

## 使用方法

1. 複製專案到本地：
```bash
git clone https://github.com/AntingZ21709/5114056012_HW2
```

2. 進入專案目錄：
```bash
cd [專案目錄]
```

3. 執行應用程式：
```bash
streamlit run app.py
```

4. 在瀏覽器中開啟顯示的網址（通常是 http://localhost:8501）

## 資料集說明

使用 Kaggle 上的「Red Wine Quality」資料集，包含 1,599 筆紅酒樣本，每筆資料包含 11 個物理化學特性指標和一個品質評分。

## 分析流程

1. 商業理解：定義問題與目標
2. 資料理解：探索資料特性
3. 資料準備：清理與預處理
4. 建模：訓練多元線性回歸模型
5. 評估：計算模型效能指標
6. 部署：使用 Streamlit 部署成網頁應用

## 作者

5114056012 鄭安婷

## 授權

本專案採用 MIT 授權。