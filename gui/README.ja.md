[中文](./README.md) | [English](./README.en.md) | [日本語](./README.ja.md) | [한국어](./README.ko.md)

# Musubi Tuner GUI

musubi-tunerの完全なワークフローを管理する、NiceGUIベースのグラフィカルインターフェース。

## 機能

- 🎨 **フルワークフロー対応**: データセットタグ付け → キャッシュ → 学習 → 推論
- 🤖 **マルチアーキテクチャ対応**: FLUX.2、Wan2.1、HunyuanVideo、FramePack、Long-CAT、Z-Image、Qwen Image、HV 1.5、Lens、Ideogram-4、HiDream O1、FLUX Kontext、Krea-2 など
- 💾 **プリセット管理**: よく使う設定の保存と読み込み
- 📝 **リアルタイムログ**: コマンド出力と進捗の確認
- 🌐 **クロスプラットフォーム**: Windows/Linux対応、ローカルまたはクラウドで実行可能
- ⚡ **直接呼び出し**: PowerShellに依存せず、Pythonスクリプトを直接呼び出し
- 🌓 **テーマ切替**: ダーク/ライトテーマ対応、設定は自動保存
- 🌐 **国際化 (i18n)**: 中国語、英語、日本語、韓国語の4言語対応
- 🧪 **高度な学習**: SOAR補助学習、D-OPSD蒸留などに対応

## インストール

```bash
# プロジェクトディレクトリに移動
cd musubi-tuner-scripts

# プロジェクトとGUIの依存関係をインストール
uv sync --extra cu130 --extra gui --extra lycoris --extra attention --index-strategy unsafe-best-match

# musubi-tunerの全依存関係（torch、accelerateなど）がインストールされていることを確認
```

## 使い方

### 方法 1: ルートディレクトリの起動スクリプト

```powershell
# プロジェクトルートで実行
./1.6.GUI.ps1

# ポート指定
./1.6.GUI.ps1 -Port 8888

# クラウドモード（外部アクセスを許可）
./1.6.GUI.ps1 -Cloud

# ネイティブウィンドウモード
./1.6.GUI.ps1 -Native

# ブラウザを自動で開かない
./1.6.GUI.ps1 -NoBrowser
```

### 方法 2: Pythonを直接実行

```bash
# プロジェクトルートで実行
python gui/launch.py

# クラウドモード（外部アクセスを許可）
python gui/launch.py --cloud

# ポート指定
python gui/launch.py --port 8888

# ネイティブウィンドウモード
python gui/launch.py --native

# ブラウザを自動で開かない
python gui/launch.py --no-browser
```

### 方法 3: gui ディレクトリから実行

```bash
cd gui
python launch.py
```

## ワークフロー

1. **データセットタグ付け** (`/tagging`)
   - Qwen-VLなどのタグ付けモデルに対応
   - 画像の一括処理
   - プロンプトの接頭辞・接尾辞のカスタマイズ

2. **キャッシュ処理** (`/cache`)
   - モデルアーキテクチャの選択
   - モデルパスの設定
   - LatentとText Encoderの出力を事前計算
   - `python -m musubi_tuner.xxx_cache_latents` を直接呼び出し

3. **LoRA学習** (`/train`)
   - マルチタブでパラメータを整理
   - 基本設定、モデルパス、学習パラメータ、ネットワーク構造、オプティマイザ、詳細オプション
   - 学習ログをリアルタイム表示
   - プリセットの保存/読み込み対応
   - `python -m accelerate.commands.launch musubi_tuner.xxx_train_network` を直接呼び出し

4. **推論・生成** (`/generate`)
   - 学習済みLoRAウェイトを使用
   - 生成パラメータの調整
   - 参照画像編集に対応
   - `python -m musubi_tuner.xxx_generate` を直接呼び出し

## 呼び出し方法

GUIはPowerShellスクリプトに依存せず、Pythonモジュールを直接呼び出します：

```bash
# Latentキャッシュ
python -m musubi_tuner.flux_2_cache_latents --dataset_config=... --vae=...

# Text Encoderキャッシュ
python -m musubi_tuner.flux_2_cache_text_encoder_outputs --dataset_config=... --text_encoder=...

# 学習（accelerateを使用）
python -m accelerate.commands.launch --mixed_precision=bf16 musubi_tuner.flux_2_train_network --dit=... --vae=...

# 推論
python -m musubi_tuner.flux_2_generate_image --dit=... --prompt=...
```

## 対応モデルアーキテクチャ

| アーキテクチャ | キャッシュモジュール | 学習モジュール | 生成モジュール |
|------|---------|---------|---------|
| FLUX.2 | flux_2_cache_latents | flux_2_train_network | flux_2_generate_image |
| FLUX Kontext | flux_kontext_cache_latents | flux_kontext_train_network | flux_kontext_generate_image |
| Wan2.1 | wan_cache_latents | wan_train_network | wan_generate_video |
| HunyuanVideo | cache_latents | hv_train_network | hv_generate_video |
| FramePack | fpack_cache_latents | fpack_train_network | fpack_generate_video |
| Long-CAT | longcat_cache_latents | longcat_train_network | - |
| Z-Image | zimage_cache_latents | zimage_train_network | zimage_generate_image |
| HV 1.5 | hv_1_5_cache_latents | hv_1_5_train_network | hv_1_5_generate_video |
| Qwen Image | qwen_image_cache_latents | qwen_image_train_network | qwen_image_generate |
| Lens | lens_cache_latents | lens_train_network | lens_generate_image |
| Ideogram-4 | ideogram4_cache_latents | ideogram4_train_network | ideogram4_generate_image |
| HiDream O1 | hidream_o1_cache_pixel | hidream_o1_train_network | hidream_o1_generate_image |
| Krea-2 | krea2_cache_latents | krea2_train_network | krea2_generate_image |

## プリセット

`gui/presets/` ディレクトリはステージごとにサブディレクトリに分かれており、それぞれTOMLプリセットファイルが含まれています：

### `presets/cache/` - キャッシュプリセット

flux2, flux_kontext, framepack, hidream_o1, hunyuan_video, hv_1_5, ideogram4, krea2, lens, long_cat, qwen_image, wan2_1, zimage, zimage_dopsd

### `presets/train/` - 学習プリセット

flux2, flux_kontext, framepack, hidream_o1, hidream_o1_dev, hunyuan_video, hv_1_5, ideogram4, krea2, lens, lens_finetune, lens_finetune_low_vram, lens_low_vram, long_cat, qwen_image, qwen_image_finetune, wan2_1, zimage, zimage_dopsd, zimage_dopsd_finetune, zimage_finetune

### `presets/generate/` - 生成プリセット

flux2, flux_kontext, framepack, hidream_o1, hidream_o1_dev_edit_flow, hidream_o1_dev_flash, hunyuan_video, hv_1_5, ideogram4, krea2, lens, long_cat, qwen_image, wan2_1, zimage

### `presets/user/` - ユーザーカスタムプリセット

GUIから保存したカスタムプリセットはこのディレクトリに保存されます。

## プロジェクト構造

```
gui/
├── main.py              # メインエントリポイント
├── launch.py            # 起動スクリプト
├── README.md            # 中国語ドキュメント
├── README.en.md         # 英語ドキュメント
├── README.ja.md         # 日本語ドキュメント（このファイル）
├── PARAMETERS.md        # パラメータマッピングドキュメント
├── UPDATES.md           # 更新履歴
├── theme.py             # テーマシステム（sd-scriptsスタイル統合）
├── STYLES_REUSE.md      # スタイル再利用ガイド
├── components/          # 再利用可能コンポーネント
│   ├── path_selector.py    # パスセレクタ
│   ├── log_viewer.py       # ログビューア
│   ├── preset_manager.py   # プリセットマネージャ
│   ├── model_selector.py   # モデルセレクタ
│   └── side_tools.py       # サイドツールバー
├── wizard/             # ウィザードステップ
│   ├── step0_setup.py      # 環境チェック
│   ├── step1_tagging.py    # データセットタグ付け
│   ├── step2_cache.py      # キャッシュ処理
│   ├── step3_train.py      # 学習
│   ├── step4_generate.py   # 推論・生成
│   ├── step7_settings.py   # 設定ページ
│   └── console_page.py     # コンソールページ
├── utils/              # ユーティリティ
│   ├── config_manager.py   # 設定管理
│   ├── process_runner.py   # プロセス実行（Python直接呼び出し）
│   ├── model_catalog.py    # モデルアーキテクチャカタログ
│   ├── port_utils.py       # ポート解決
│   └── i18n.py             # 国際化（sd-scriptsから再利用）
├── presets/            # プリセット設定
│   ├── cache/              # キャッシュプリセット (*.toml)
│   ├── train/              # 学習プリセット (*.toml)
│   ├── generate/           # 生成プリセット (*.toml)
│   └── user/               # ユーザーカスタムプリセット
└── examples/           # 使用例
    └── reuse_styles_example.py
```

## 注意事項

1. **作業ディレクトリ**: GUIはデフォルトでプロジェクトルートからスクリプトを実行します。パスが正しく設定されていることを確認してください
2. **依存関係**: musubi-tunerの全依存関係（torch、accelerateなど）がインストールされている必要があります
3. **VRAM**: モデルと設定によっては、大きなVRAMが必要な場合があります
4. **プリセット**: プリセットはパラメータのみを保存し、モデルパスは保存しません。実際の環境に合わせて調整してください
5. **Pythonモジュール**: `musubi-tuner`ディレクトリがPythonパスに含まれているか、パッケージとしてインストールされていることを確認してください

## FAQ

**Q: 新しいモデルアーキテクチャのサポートを追加するには？**
A: `gui/utils/model_catalog.py` を編集し、`MODEL_CATALOG` ディクショナリに新しいアーキテクチャを追加してください。

**Q: 学習スクリプトのパラメータをカスタマイズするには？**
A: 対応するステップページのコード（例：`step3_train.py`）で、パラメータ構築ロジックを変更してください。

**Q: クラウドサーバーで使用するには？**
A: `--cloud` パラメータを指定して起動し、ブラウザで `http://<サーバーIP>:7788`（デフォルトポート）にアクセスしてください

**Q: カスタム設定を保存するには？**
A: 学習ページの「プリセットとして保存」ボタンをクリックし、名前を入力してください。

**Q: モジュールが見つからないというエラーが出る？**
A: プロジェクトルートから実行し、`musubi-tuner` ディレクトリが存在することを確認してください。以下を試すこともできます：
```python
import sys
sys.path.insert(0, '.')
```

## テーマと国際化

### ダーク/ライトテーマ

GUIはダークとライトの両方のテーマに対応しています。右上の太陽/月のアイコンをクリックして切り替えます。テーマ設定はブラウザの `localStorage` に保存され、次回アクセス時に自動的に読み込まれます。

### Modern Theme（デフォルト）
- ダーク背景、モダンなカードとボタン
- ディープグリーン＋ゴールドのナチュラルカラースキーム

### Green Gold Theme（sd-scriptsから）
- 明るい背景、伝統的なグリーンゴールドカラースキーム
- `sd-scripts/gui/styles.py` から再利用

### 国際化 (i18n)

4言語（中国語、英語、日本語、韓国語）に対応。右上の言語ドロップダウンをクリックして切り替えます。切り替え後、ページが自動的にリフレッシュされて新しい言語が適用されます。

詳細は `STYLES_REUSE.md`、`UPDATES.md`、`examples/reuse_styles_example.py` を参照してください。

## ライセンス

musubi-tunerメインプロジェクトと同じです。
