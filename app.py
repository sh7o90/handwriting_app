# 必要なライブラリをインポート
import streamlit as st  # Streamlitライブラリ
from streamlit_drawable_canvas import st_canvas  # 描画キャンバス用ライブラリ
from PIL import Image, ImageColor  # 画像処理用ライブラリ
import torch  # PyTorchライブラリ
import torchvision.transforms as transforms  # 画像変換処理用ライブラリ
import torch.nn as nn  # PyTorchのニューラルネットワークライブラリ
import timm  # PyTorch Image Models (timm)ライブラリ
import pickle  # ピクルスファイル読み込み用ライブラリ
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

# 画像をテンソルに変換するための変換パイプライン
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # グレースケール変換
    transforms.Resize((224, 224)),  # 画像を224x224ピクセルにリサイズ
    transforms.ToTensor(),  # テンソルに変換
    transforms.Normalize(mean=[0.1307], std=[0.3081])  # 正規化
])

# ラベルのリスト(EMNISTのクラスに合わせる)
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + \
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z'] + \
        ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

def get_label(class_id):
    return labels[class_id]

# 画像を前処理する関数
def preprocess_image(image):
    image_gray = image[:, :, 3]
    image_pil = Image.fromarray(image_gray)
    image_resize = image_pil.resize((28, 28))
    image_tensor = transform(image_resize).unsqueeze(0)
    return image_tensor

# EfficientNetV2_MNISTモデルの定義
class EfficientNetV2_EMNIST(nn.Module):
    def __init__(self, num_classes=47):
        super().__init__()
        self.efficientnet = timm.create_model('tf_efficientnetv2_b0', pretrained=True)
        # 最終層の出力数を変更
        in_features = self.efficientnet.classifier.in_features
        self.efficientnet.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # 入力画像のチャンネル数を3に変更し、(224, 224, 3) の形状に変更
        x = x[:, :3, :, :]
        return self.efficientnet(x)

# Streamlitアプリのタイトル
st.title("手書き文字認識アプリ")

# 白色の背景画像を作成
width, height = 300, 300  # キャンバスのサイズに合わせる
background_image = Image.new('RGB', (width, height), color=ImageColor.getrgb('white'))

# レイアウトの調整
col1, col2 = st.columns([2, 1])  # キャンバスと予測ボタンの列を作成

# Canvas領域の作成
with col1:
    st.subheader("文字を書いてください")
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#000",
        background_image=background_image,
        update_streamlit=True,
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True,  # ツールバーを表示
    )

# モデルのロード
net = EfficientNetV2_EMNIST()  # EfficientNetV2_MNISTモデルのインスタンス化
with open('last_efficientnetv2_emnist.pkl', 'rb') as f:
    net.load_state_dict(pickle.load(f))  # 学習済みモデルの読み込み
net.eval()  # 推論モードに設定

# 予測ボタンが押されたときの処理
with col2:
    st.subheader("予測結果")
    predict_button = st.button("Predict", key="predict_button")
    if predict_button:
        with st.spinner("予測中..."):  # ローディングアニメーションを追加
            # 描画された画像データを取得し、RGBA形式からRGB形式に変換
            image_data = canvas_result.image_data
            # 画像を前処理
            processed_image = preprocess_image(image_data)

            # 予測
            with torch.no_grad():
                outputs = net(processed_image)

            probabilities = torch.softmax(outputs, dim=1)

            # 確率が高い順に並び替え
            sorted_probs, sorted_indices = torch.sort(probabilities, dim=1, descending=True)

            # 予測結果を大きい文字で表示
            result_text = f"結果: {get_label(sorted_indices[0, 0].item())}"
            st.markdown(f"<h1 style='text-align: center; color: Text color;'>{result_text}</h1>", unsafe_allow_html=True)

            # 上位5つのクラスとその確率を取得
            labels = [get_label(idx) for idx in sorted_indices[0, :5]]
            values = [sorted_probs[0, i].item() for i in range(5)]

with st.expander("クリックして展開"):
    # サブヘッダーを中央揃えで表示
    st.markdown(f"<h3 style='text-align: center; color: Text color;'>確率分布</h3>", unsafe_allow_html=True)
    if predict_button:
        # 棒グラフを描画
        fig, ax = plt.subplots(figsize=(8, 6))  # グラフサイズを大きくする
        ax.bar(labels, values, width=0.6)  # 棒の幅を調整
        ax.set_xlabel("クラス", fontsize=14)  # ラベルのフォントサイズを大きくする
        ax.set_ylabel("確率", fontsize=14)
        ax.tick_params(axis='x', labelsize=12)  # x軸ラベルの角度と文字サイズを調整
        ax.grid(True)  # グリッドを表示
        ax.set_axisbelow(True)  # グリッドを背面に表示

        # Streamlitで棒グラフを表示
        st.pyplot(fig)
