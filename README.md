# 手書き文字認識アプリ
このアプリは、Streamlitを使って作成された手書き文字認識アプリです。

EfficientNetV2モデルを使用し、EMNISTデータセットで学習されています。

# 機能
キャンバス上に手書きの文字を描画できます。

描画された文字を認識し、予測結果を表示します。

上位5つの予測クラスとその確率を棒グラフで表示します。

# 使用技術

* Python
* Streamlit
* PyTorch
* EfficientNetV2
* EMNIST Dataset

# 実行方法
このアプリは、GitHub上のリポジトリから入手できます。以下のコマンドを使ってリポジトリをクローンしてください。

```git clone https://github.com/sh7o90/handwriting_app```

このコマンドを実行すると、handwriting_appというディレクトリにアプリのソースコードがダウンロードされます。

クローンが完了したら、以下の手順に従ってアプリを実行してください。

１．必要なライブラリをインストールします。

```cd handwriting_app```

```pip install -r requirements.txt```

２．アプリを起動します。

```streamlit run app.py```

３．ブラウザで以下にアクセスすると、アプリが表示されます。

```http://localhost:8501```

# デモ
https://github.com/sh7o90/handwriting_app/assets/158803446/c2ed37be-002b-400c-8a4b-1a5ee91a10f3

# ライセンス
このプロジェクトは MIT ライセンスの下で公開されています。[MIT license](https://en.wikipedia.org/wiki/MIT_License)

# 作者
奥野 翔太(sh7o90)



