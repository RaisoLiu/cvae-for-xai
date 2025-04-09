# CVAE for XAI

## 專案目標

本專案旨在利用條件變分自編碼器 (Conditional Variational Autoencoder, CVAE) 探索與實現可解釋人工智慧 (Explainable AI, XAI) 的相關應用。目標包含但不限於：

*   生成特定條件下的數據樣本。
*   探索數據在潛在空間 (Latent Space) 中的表示。
*   基於潛在空間的操作生成反事實解釋 (Counterfactual Explanations)。
*   視覺化模型決策或數據特徵。

## 專案結構

```
.
├── checkpoints/          # 存放訓練好的模型檢查點
├── config/               # 存放設定檔 (模型、訓練、資料等)
├── data/                 # (可選) 存放原始數據或特定處理腳本
├── docs/                 # (可選) 專案相關文件
├── output/               # 存放實驗結果 (日誌、指標、圖像、視覺化結果)
├── scripts/              # 主要的執行腳本
│   ├── train.py          # 訓練模型
│   ├── evaluate.py       # 評估模型
│   ├── predict.py        # 使用模型進行預測
│   └── preprocess.py     # (可選) 資料預處理
├── src/                  # 主要 Python 原始碼
│   ├── data/             # 資料載入與處理模組
│   ├── models/           # 模型架構定義 (CVAE, Encoder, Decoder)
│   ├── training/         # (可選) 訓練迴圈與邏輯
│   └── utils/            # 公用函式 (日誌、評估指標等)
│   └── main.py           # (可選) 統一的程式進入點
├── tests/                # (可選) 單元測試與整合測試
├── .gitignore            # Git 忽略檔案列表
├── pyproject.toml        # Python 專案設定檔 (例如 Poetry)
└── requirements.txt      # Python 依賴套件列表 (或由 pyproject.toml 管理)
```

## 安裝與設定

1.  **複製專案庫**
    ```bash
    git clone <your-repository-url>
    cd cvae-for-xai
    ```
2.  **建立虛擬環境** (建議)
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **安裝依賴套件與專案**
    ```bash
    pip install -e .
    # 如果您需要開發依賴項 (例如運行測試):
    # pip install -e ".[dev]"
    ```

## 使用方法

### 1. 設定

*   在 `config/` 目錄下編輯或建立 YAML 設定檔 (例如 `model.yaml`, `training.yaml`, `data.yaml`) 來定義模型架構、超參數、資料路徑等。

### 2. 訓練模型

*   執行訓練腳本，並指定設定檔：
    ```bash
    python scripts/train.py --config_model config/model.yaml --config_train config/training.yaml --config_data config/data.yaml --experiment_name my_first_experiment
    ```
*   訓練過程中的日誌、輸出的模型檢查點和視覺化結果將會儲存於 `output/{實驗名稱或ID}/` 和 `checkpoints/{實驗名稱或ID}/`。實驗名稱會基於設定與時間戳自動生成，或可透過參數指定。

### 3. 評估模型

*   使用訓練好的模型檢查點進行評估：
    ```bash
    python scripts/evaluate.py --checkpoint checkpoints/{實驗名稱或ID}/checkpoint_best.pth --config_data config/data.yaml
    ```

### 4. 進行預測/生成

*   載入模型進行預測或生成新的樣本：
    ```bash
    python scripts/predict.py --checkpoint checkpoints/{實驗名稱或ID}/checkpoint_best.pth --input_data <path-to-your-data> --output_dir output/{實驗名稱或ID}/predictions/
    ```

## 實驗輸出

*   **`output/`**: 包含每次實驗運行的詳細結果。每個子資料夾代表一次獨立的實驗，其命名通常包含模型、資料集、關鍵參數和時間戳等資訊。內部包含：
    *   `config.yaml`: 該次實驗使用的設定檔副本。
    *   `train.log`: 訓練過程的詳細日誌。
    *   `metrics.json`: 最終的評估指標。
    *   `plots/`: 訓練曲線、視覺化圖表等。
    *   `visualizations/`: 重建樣本、潛在空間視覺化等。
*   **`checkpoints/`**: 存放訓練過程中儲存的模型權重 (`.pth` 或 `.pt` 檔案)，用於恢復訓練或後續的評估、預測。

