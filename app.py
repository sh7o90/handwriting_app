# 必要なライブラリをインポート
import streamlit as st  # Streamlitライブラリ
from streamlit_drawable_canvas import st_canvas  # 描画キャンバス用ライブラリ
from PIL import Image  # 画像処理用ライブラリ
import torch  # PyTorchライブラリ
import torchvision.transforms as transforms  # 画像変換処理用ライブラリ
import torch.nn as nn  # PyTorchのニューラルネットワークライブラリ
import timm  # PyTorch Image Models (timm)ライブラリ
import pickle  # ピクルスファイル読み込み用ライブラリ

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
st.title("EMNIST Character Recognition")

# レイアウトの調整
col1, col2 = st.columns([2, 1])  # キャンバスと予測ボタンの列を作成

# Canvas領域の作成
with col1:
    st.subheader("Draw a Digit")
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#000",
        update_streamlit=True,
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

# モデルのロード
net = EfficientNetV2_EMNIST()  # EfficientNetV2_MNISTモデルのインスタンス化
with open('last_efficientnetv2_emnist.pkl', 'rb') as f:
    net.load_state_dict(pickle.load(f))  # 学習済みモデルの読み込み
net.eval()  # 推論モードに設定

# 画像を前処理する関数
def preprocess_image(image):
    # アルファチャンネルだけ取得してグレースケールに変換
    image_gray = image[:, :, 3]
    # Numpy配列をPIL Imageに変換
    image_pil = Image.fromarray(image_gray)
    # 28x28ピクセルにリサイズ
    image_resize = image_pil.resize((28, 28))
    # 変換を適用してテンソルに変換
    image_tensor = transform(image_resize).unsqueeze(0)
    return image_tensor

# 予測ボタンが押されたときの処理
with col2:
    st.subheader("Prediction")
    predict_button = st.button("Predict", key="predict_button")
    if predict_button:
        with st.spinner("Predicting..."):  # ローディングアニメーションを追加
            # 描画された画像データを取得し、RGBA形式からRGB形式に変換
            image_data = canvas_result.image_data
            # 画像を前処理
            processed_image = preprocess_image(image_data)
            # 予測
            with torch.no_grad():
                outputs = net(processed_image)
                # 最も確率の高いクラスのインデックスを取得
                predicted_class = torch.argmax(outputs, dim=1).item()

            # 予測結果を表示
            st.success(f"Predicted Digit: {predicted_class}({get_label(predicted_class)})")