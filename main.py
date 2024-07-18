from google.colab import drive
drive.mount('/content/drive')

!unzip "/content/drive/MyDrive/VQA/train.zip" -d "/content/drive/MyDrive/Colab Notebooks/unzipped_data"
!unzip "//content/drive/MyDrive/VQA/valid.zip" -d "/content/drive/MyDrive/Colab Notebooks/unzipped_data"

import os
import shutil
from concurrent.futures import ThreadPoolExecutor

#コピー元とコピー先のディレクトリを指定
src_dir = '/content/drive/MyDrive/Colab Notebooks/unzipped_data'
dst_dir = '/content/data'

#コビー先ディレクトリを作成
os.makedirs(dst_dir, exist_ok=True)

#コピーするファイルのリストを作成
files_to_copy = []
for root, _, files in os.walk(src_dir):
    for file in files:
        src_file = os.path.join(root, file)
        dst_file = os.path.join(dst_dir, os.path.relpath(src_file, src_dir))
        files_to_copy.append((src_file, dst_file))

# 並列でファイルをコピーする関数
def copy_file(src_dst):
    src_file, dst_file = src_dst
    dst_file_dir = os.path.dirname(dst_file)
    os.makedirs(dst_file_dir, exist_ok=True)
    try:
        shutil.copy2(src_file, dst_file)
    except Exception as e:
        print(f"Error copying {src_file} to {dst_file}: {e}")

# ThreadPoolExecutorを使って並列でコピー
with ThreadPoolExecutor(max_workers=24) as executor:
    executor.map(copy_file, files_to_copy)

#ファイルがすべてコピーされたかを確認
copied_files = [os.path.join(root, file) for root, _, files in os.walk(dst_dir) for file in files]
missing_files = [src for src, dst in files_to_copy if dst not in copied_files]

if missing_files:
    print(f"Missing files: {missing_files}")
else:
    print( "All files copied successfully.")

import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.optim import lr_scheduler # 追記部分


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    # lowercase
    text = text.lower() # str(文字列)に対して定義されているメソッド。文字列中の全ての文字を小文字に変換する。

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True): # initはインスタンス化したときに実行される処理
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        # question / answerの辞書を作成  # 質問文→各文に対するidラベルの割り当て、のための辞書の用意

        # コーパスによる辞書の拡張（追記部分）
        df_class_mapping = pandas.read_csv("https://huggingface.co/spaces/CVPR/VizWiz-CLIP-VQA/raw/main/data/annotations/class_mapping.csv")
        class_dic = dict(zip(df_class_mapping["answer"], df_class_mapping["class_id"]))
        """
        self.question2idx = {}
        self.idx2question = {}
        """
        self.answer2idx = class_dic # 追記部分　なぜquestionには追加しないのか←questionの辞書を大きくしても、訓練で使う単語（訓練データ内の単語数）はそのままだから？、テストデータのquestionはカバーできるのかという疑問があるが、そもそもテストデータには答えがないため質問文のペアがいない
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        self.question = self.df["question"]# 追記部分
        """
        for question in self.df["question"]:
            question = process_text(question)

            words = question.split(" ")  # 質問文を単語単位に分解
            for word in words:
                if word not in self.question2idx:   # 辞書に追加されていない単語があれば、辞書{word:id}のwordに、新しく追加されたときの順番をidラベルとして割り当てる。
                    self.question2idx[word] = len(self.question2idx) # この時点では、辞書は文字列wordと数値idのペアの集合
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)　#　を上記で作成した辞書から作成
        """
        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset): # インスタンス化ではなく、インスタンスのメソッドを明示的に呼び出したときに動作する処理
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """

        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer
        """
        self.question2idx = dataset.question2idx # selfはインスタンス自身。このコードでは、作成されたインスタンスに対してquestion2idxメソッドを定義している。で、その定義する内容は
        self.idx2question = dataset.idx2question
        """

    def __getitem__(self, idx):  # getitemはobject[]のようにインスタンスに角括弧を指定したときに呼び出される処理を定義
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}") # 文字列 " ～/～"が最終的な名前であるファイルを読み込む
        image = self.transform(image)
        question = self.question[idx] # 追記部分
        """
        question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加　# 辞書に登録されている単語数だけの0を並べる
        question_words = self.df["question"][idx].split(" ")   # df["question"][idx]は、questionカラムからidx番目のレコードを取り出すという意味
        for word in question_words:
            try:
                question[self.question2idx[word]] = 1  # one-hot表現に変換 # self.question2idx[word]: question_words（あるquestion内のワード）内の指定したwordのid(番号)を取得。idによって1にする成分を指定し、1を割り当て。
            except KeyError:
                question[-1] = 1  # 未知語
        """
        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]  # 10個の回答文を格納したanswersのセルから、各回答文answerを取り出し、辞書?のキーanswerを選択して実際の回答文を取り出す
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, question, torch.Tensor(answers), int(mode_answer_idx) # 全てのデータが数値型になった、mode_answer_idxには多分、最頻値の回答には1、それ以外は0というようなダミーがついている。torch.Tensor(question)

        else:
            return image, question # torch.Tensor(question)

    def __len__(self):
        return len(self.df)

# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:  # まず上のfor文からi=0を取り出して固定し、jだけを変えていく。総当たりi vs jを階層的なfor文で行う
                    continue
                if pred == answers[j]: # 予測した文と回答文が一致した場合
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)

# 3. モデルの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        #　以下はモデル内の各モジュールの実装（定義）であって、順伝播の流れはまだ作られていない。順伝播はここで定義した各モジュールを使って組み立てられる。
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual) # 残差接続（∈スキップ接続）
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 特徴を強く表すところを抽出する

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])  # 引数については、ResNet(block, layers)


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3]) # 同じくResNet(block, layers)

"""
class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int):
        super().__init__()
        self.resnet = ResNet18()
        self.text_encoder = nn.Linear(vocab_size, 512) # nn.Linearの段階では引数は入出力のベクトルの次元数、インスタンス化するとデータを引数とする

        self.fc = nn.Sequential(  # 全結合層 (fully-connected layer)
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    # 「画像処理」と「自然言語処理」の接続点、順伝播。ここもいじらない方がよい。
    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像の特徴量
        question_feature = self.text_encoder(question)  # テキストの特徴量　# one-hotベクトルのqustionを入力

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x
"""

from transformers import BertModel, BertTokenizer

class VQAModel(nn.Module):
    def __init__(self, n_answer: int):
        super().__init__()

        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained(
            "bert-base-uncased", torch_dtype=torch.float32, attn_implementation="sdpa"
        )
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.resnet = ResNet18()
        self.fc = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer),
        )

    def forward(self, image, question):
        N = image.shape[0]
        image_feature = self.resnet(image)
        assert image_feature.shape == (N, 512)
        with torch.no_grad():
            question = self.bert_tokenizer(
                question,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(image.device)
            question_feature = self.bert_model(**question).last_hidden_state[:, 0, :]  # (N, 768)
            assert question_feature.shape == (N, 768)
        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)
        x = torch.nn.functional.softmax(x, dim=1)

        return x

# 4. 学習の実装 # このセルはいじらない方がよい
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer = image.to(device, non_blocking=True), question, answers.to(device, non_blocking=True), mode_answer.to(device, non_blocking=True)
        """
        image, question, answer, mode_answer = \
            image.to(device), question, answers.to(device), mode_answer.to(device) # question.to(device)
        """

        pred = model(image, question)  # モデルの最終的な出力
        loss = criterion(pred, mode_answer.squeeze()) # モデルの最終的な出力と、正解ラベル（ダミー変数）を使って損失を測る

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device): #
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question, answers.to(device), mode_answer.to(device) # question.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloader / model
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    """
    # 追記部分
    class gcn():
        def __init__(self):
            pass

        def __call__(self, x):
            mean = torch.mean(x)
            std = torch.std(x)
            return (x - mean)/(std + 10**(-6))  # 0除算を防ぐ

    GCN = gcn()

    transform_test = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), GCN])

    transform_rotate = transforms.Compose([transforms.RandomRotation(degrees=(-10, 10)), transforms.Resize(size=(224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), GCN])
    transform_center = transforms.Compose([transforms.CenterCrop(size=(20, 20)), transforms.Resize(size=(224,224)),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), GCN])
    transform_normal = transforms.Compose([transforms.Resize(size=(224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),GCN])
    transform_gauss = transforms.Compose([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),transforms.Resize(size=(224,224)),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), GCN])

    transform_train = transforms.RandomChoice([transform_rotate, transform_center, transform_normal, transform_gauss])
    #　以上
    # /content/drive/MyDrive/Colab Notebooks/unzipped_data/train

    train_dataset = VQADataset(df_path="/content/drive/MyDrive/VQA/train.json", image_dir="/content/data/train", transform=transform_train) # ./data/train.json #./data/train
    test_dataset = VQADataset(df_path="/content/drive/MyDrive/VQA/valid.json", image_dir="/content/data/valid", transform=transform_test, answer=False)  # ./data/valid.json # ./data/valid
    test_dataset.update_dict(train_dataset) # たぶんテストデータにVQAdatasetという前処理だけ施しておいて、辞書に関してのみテストデータから訓練データのものに上書きすることで、辞書を統一する。テスト用の辞書を作る

# 学習の実行
import os # 追記部分

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=os.cpu_count() ,pin_memory=True) # 追記部分 num_worker, pin_memory
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True) # 追記部分

model = VQAModel(n_answer=len(train_dataset.answer2idx)).to(device) # 特定のquestion内の単語数+1をvocab_sizeへ、実際のanswer内の単語数をn_answerへ 。 vocab_size=len(train_dataset.question2idx)+1

# optimizer / criterion
num_epoch = 5
criterion = nn.CrossEntropyLoss() # 損失関数はクロスエントロピー
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95) # 追記部分

# train model
for epoch in range(num_epoch):
    train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
    print(f"【{epoch + 1}/{num_epoch}】\n"
          f"train time: {train_time:.2f} [s]\n"
          f"train loss: {train_loss:.4f}\n"
          f"train acc: {train_acc:.4f}\n"
          f"train simple acc: {train_simple_acc:.4f}")
    scheduler.step() # 追記部分

# 提出用ファイルの作成
model.eval()
submission = []
for image, question in test_loader:
    image = image.to(device)
    pred = model(image, question)
    pred = pred.argmax(1).cpu().item() # 予測値の中から（多分、確率の）最大値となるidxをとる、辞書に収録された全ての単語に対して生起確率が出力される？
    submission.append(pred)

submission = [train_dataset.idx2answer[id] for id in submission] # 辞書を使って、予測をidxから単語に変換して、予測文をつくる
submission = np.array(submission)
torch.save(model.state_dict(), "model.pth")
np.save("submission.npy", submission)

if __name__ == "__main__":
    main()
