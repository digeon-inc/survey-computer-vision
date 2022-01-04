# survey-computer-vision

## textlint

```shell
$ yarn install
$ npx textlint README.md
```

## ResNet

### 論文概要

#### どんなもの？
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Microsoft Researchが2015年に提案したモデル。

#### 先行研究と比べてどこがすごい？

当時、画像認識において一般にCNNの層数を増やすことでより高次元の特徴を獲得することは知られていたが、単純に層を重ねるだけでは性能が悪化していく勾配消失問題があった。ResNetでは`shortcut connection`という機構を導入し、手前の層の入力を後ろの層に直接足し合わせることで、この勾配消失問題を解決した。

#### 技術や手法のキモはどこ？

![](./images/resnet_shortcut-connection.png)

<p style="text-align: center;">画像は<a href="https://arxiv.org/pdf/1512.03385.pdf" target="blank_">論文</a>より引用</p>

図の左側が`building block`と呼ばれ、右側が`bottleneck building block`と呼ばれる。この構造によって勾配がより手前の層まで伝わるようになった。

#### どうやって有効だと検証した？
ILSVRCという毎年開催されていたImageNetを用いた画像分類コンペにおいて、2014年以前はせいぜい20層程度(VGGが16か19層，GoogleNetが22層)のモデルで競っていた。しかしResNetは152もの層を重ねて学習させることに成功し、2015年のILSVCRで優勝した。

#### 議論はある？

#### 次に読むべき論文は？

### アーキテクチャ詳細

### 実装の参考

- [vision/resnet.py at main · pytorch/vision](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
