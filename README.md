# survey-computer-vision

## textlint

```shell
$ yarn install
$ npx textlint README.md
```

## AlexNet

### 論文概要

#### どんなもの？
[ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

2012年のILSVRCというImageNetを用いた画像分類コンペで圧勝したモデル。

#### 先行研究と比べてどこがすごい？
2012年までは人間が画像から特徴量を設計し、それを用いて分類を行っていたが、AlexNetが2012年のILSVRCで圧勝したことにより機械によって特徴量を抽出できることが示された。また当時、畳み込み演算を含む大規模なCNNを学習することはコストの面で難しかったが、GPUに最適化された畳み込み演算の実装を行うことで学習時間を短縮した。

#### 技術や手法のキモはどこ？
- ReLU
  - <img src="https://latex.codecogs.com/svg.image?f(x)=max(0,&space;x)" title="f(x)=max(0, x)" />
  - シグモイド関数やtanh関数などの非線形関数よりも数倍速く学習できる
  - 勾配消失対策
- GPUによる並列計算
  - ネットワークの中の一部のみでGPU間の通信を行うことで、計算を効率化した。
- ReLUと輝度の正規化の組み合わせが良かった。
- Overlapping Pooling
  - プーリング層でダウンサンプルする領域を少しずつ被せることで過学習しにくくなった
- Dropoutで過学習を防いだ。
- 水平移動や反転などのData Argumentationによって教師データを増やした。

#### どうやって有効だと検証した？
ImageNetの2011年秋の版で事前学習したAlexNetを2012年のILSVRCでfine tuningすると、テストデータにおいてエラー率は15.3%となった。

#### 議論はある？
中間層をひとつでも取り除くとネットワークの性能が低下するので、ネットワークの深さは非常に重要である。教師データを増やさずにネットワークを大きくしたり、教師なしの事前学習でさらなる性能が得られるかもしれない。

#### 次に読むべき論文は？

### アーキテクチャ詳細
![](./images/alexnet.png)

<p style="text-align: center;">画像は<a href="https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf" target="blank_">論文</a>より引用</p>

5層の畳み込み層と3層の全結合層からなる。

### 実装の参考
- [vision/alexnet.py at main · pytorch/vision](https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py)

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
ILSVRCという毎年開催されていたImageNetを用いた画像分類コンペにおいて、2014年以前はせいぜい20層程度(VGGが16か19層、GoogleNetが22層)のモデルで競っていた。しかしResNetは152もの層を重ねて学習させることに成功し、2015年のILSVCRで優勝した。

#### 議論はある？

#### 次に読むべき論文は？

### アーキテクチャ詳細

### 実装の参考

- [vision/resnet.py at main · pytorch/vision](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

## Transformer

### 論文概要

#### どんなもの？
[Attention is All You Need](https://arxiv.org/abs/1706.03762)

2017年に提案された自然言語処理のモデル。コンピュータビジョンのモデルではないものの、この論文で提案されたTransformerというアーキテクチャが後発のコンピュータビジョンのモデルに大きな影響を与えたので、ここで取り上げる。

#### 先行研究と比べてどこがすごい？
当時、Seq2SeqのモデルではRNNやCNNと併用してAttentionを用いるものがあった。しかし、この論文ではRNNやCNNを排除してAttentionのみのTransformerというアーキテクチャを提案した。RNNを排除してAttentionや全結合層を用いることによって並列化が可能になって学習にかかる時間が削減された。さらに精度も向上し、入力と出力の文章離れた位置にある任意の依存関係を学習しやすくなった。

#### 技術や手法のキモはどこ？

- Scaled Dot Product Attention
  - まず、embeddingされた入力ベクトル(sequence_length, d_model)に対して、Query、Key、Valueを計算する。これらは入力ベクトルをそれぞれ重み行列Wq(d_model, d_q)、Wk(d_model, d_k)、Wv(d_model, d_v)で写像して得られる。この論文ではself-attentionを用いるのと、全く同じ構造のレイヤーを複数重ねるので、d_q、d_k、d_v、d_modelは全て等しい。
  - 各Queryと各Keyの内積にsoftmax関数を適用し、QueryとKeyの関連度を計算する。さらにこれをValueとの内積をとることで、各Queryと類似度の高いKeyに対応するValueほど重く重みづけされたValueの重み付き和が得られる。これは、入力のある部分に対して他のどの部分が重要になるかを抽出する操作とみなすことができる。
  - <img src="https://latex.codecogs.com/svg.image?Attention(Q,&space;K,&space;V)&space;=&space;softmax(\frac{QK^T}{\sqrt{d_k}})V" title="Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V" />
  - QueryとKeyの内積を<img src="https://latex.codecogs.com/svg.image?\sqrt{d_k}" title="\sqrt{d_k}" />で割っているのは、<img src="https://latex.codecogs.com/svg.image?d_k" title="d_k" />が大きくなったときにQueryとKeyの内積が大きくなり、softmaxの勾配が極端に小さくなっていまうのを防ぐためである。
  
Scaled Dot Product AttentionはPyTorchを用いて次のように実装できる。  
```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        """
        Args:
            d_model: embedded vector length
        """
        super().__init__()
        self.d_k: int = d_model
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        Args:
            x: input tensor [batch_size, sequence_length, d_model]
        Returns:
            out: torch.Tensor [batch_size, sequence_length, d_model]
        """
        query, key, value = self.w_q(x), self.w_k(x), self.w_v(x)
        query /= math.sqrt(self.d_k)
        attn_weight = torch.bmm(query, key.transpose(-2, -1))
        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask, -1e9)
        attn_weight = functional.softmax(attn_weight, dim=-1)
        attn = torch.matmul(attn_weight, value)
        out = self.out(attn)
        return out
```
- Multi Head Attention
  - 各単語に対して1組の<img src="https://latex.codecogs.com/svg.image?d_{model}" title="d_{model}" />次元のQuery、Key、Valueを持たせるのではなく、それらを<img src="https://latex.codecogs.com/svg.image?h" title="h" />種類の異なる重みベクトルにより写像したもので<img src="https://latex.codecogs.com/svg.image?h" title="h" />種類の異なるAttentionを計算する。それぞれのQuery、Key、Valueから構成されるAttention機構はheadと呼ばれ、それぞれで異なる部分空間から有益な情報を抽出することができる。 これによって、アンサンブルのような効果が得られる
  - <img src="https://latex.codecogs.com/svg.image?\begin{aligned}Muiltihead(Q,K,V)&=Concat(head_1,&space;...,&space;head_h)W^O\\where\quad&space;head_i&=Attention(QW^i_Q,KW^i_K,VW^i_V)\end{aligned}" title="\begin{aligned}Muiltihead(Q,K,V)&=Concat(head_1, ..., head_h)W^O\\where\quad head_i&=Attention(QW^i_Q,KW^i_K,VW^i_V)\end{aligned}" />  
  ただし<img src="https://latex.codecogs.com/svg.image?W^i_Q\in\mathbb{R}^{d_{model}\times&space;d_k},W^i_K\in\mathbb{R}^{d_{model}\times&space;d_k},W^i_V\in\mathbb{R}^{d_{model}\times&space;d_v}" title="W^i_Q\in\mathbb{R}^{d_{model}\times d_k},W^i_K\in\mathbb{R}^{d_{model}\times d_k},W^i_V\in\mathbb{R}^{d_{model}\times d_v}" />  

 
Multi Head Attentionは、Scaled Dot Product Attentionの実装を少し改変する事で実装できる。計算効率のため、各headに分割されたQuery, Key, Valueをそれぞれまとめて保持する。Query、Key、Valueのテンソルの形状は(batch_size, num_head, sequence_length, dim_per_head)のようになる。
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_head: int) -> None:
        """
        Args:
            d_model: embedded vector length
        """
        super().__init__()
        assert d_model % num_head == 0, f"d_model({d_model}) must be dividible by num_head({num_head})"
        self.d_k: int = d_model
        self.num_head: int = num_head
        self.dim_per_head: int = d_model // num_head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor [batch_size, sequence_length, d_model]
        Returns:
            out: torch.Tensor [batch_size, sequence_length, d_model]
        """
        batch_size: int = x.size()[0]
        query: torch.Tensor = self.w_q(x)
        key: torch.Tensor = self.w_k(x)
        value: torch.Tensor = self.w_v(x)
        # [batch_size, sequence_length, d_model] -> [batch_size, sequence_length, num_head, dim_per_head] -> [batch_size, num_head, sequence_length, dim_per_head]
        query = query.view(batch_size, -1, self.num_head, self.dim_per_head).transpose(1,2)
        key = key.view(batch_size, -1, self.num_head, self.dim_per_head).transpose(1,2)
        value = value.view(batch_size, -1, self.num_head, self.dim_per_head).transpose(1,2)
        query /= math.sqrt(self.d_k)
        attn_weight = torch.matmul(query, key.transpose(2, 3))
        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask, -1e9)
        attn_weight = functional.softmax(attn_weight, dim=-1)
        attn = torch.matmul(attn_weight, value)
        # [batch_size, num_head, sequence_length, dim_per_head] -> [batch_size, sequence_length, num_head, dim_per_head] -> [batch_size, sequence_length, d_model]
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k)
        out = self.out(attn)
        return out
```
- Positional Encoding
  - 再帰や畳み込みを排除しているので、単語の順序把握するためにPositional Encodingを利用する。
  - ここではサインとコサインを用い、posを単語の位置、iをd_model、すなわち入力ベクトルの次元として以下のような式で計算する。計算した値は入力ベクトルに単に加算される。
  - <img src="https://latex.codecogs.com/svg.image?\begin{aligned}PE_{(pos,2i)}&=sin(pos/10000^{2i/d_{model}})\\PE_{(pos,2i&plus;1)}&=cos(pos/10000^{2i/d_{model}})\end{aligned}&space;" title="\begin{aligned}PE_{(pos,2i)}&=sin(pos/10000^{2i/d_{model}})\\PE_{(pos,2i+1)}&=cos(pos/10000^{2i/d_{model}})\end{aligned}" /> 
  - Transformerはサインやコサインといった関数の形状を簡単に学習できるため、位置エンコーディングとしてこのような関数を利用している。

#### どうやって有効だと検証した？
WMT 2014 English-German、 WMT 2014 English-Frenchというデータセットで英語からドイツ語、フランス語への翻訳タスクを行った。結果は英語-ドイツ語翻訳タスクではBLEUスコア28.4、英語-フランス語翻訳タスクではBLEUスコア41.8でstate-of-the-artを達成した。

#### 議論はある？
Attentionベースのモデルに期待しており、これをテキスト以外に画像、動画、音声などで活用できるようなAttentionを研究する予定だ。

#### 次に読むべき論文は？


### アーキテクチャ詳細
![](./images/transformer.png)

<p style="text-align: center;">画像は<a href="https://arxiv.org/pdf/1706.03762.pdf" target="blank_">論文</a>より引用</p>

- DecoderのMasked Multi Head Attentionは、対象単語より左の単語のみに依存するように、softmaxの入力値を一部マスクしている。
- Multi Head AttentionやFeed ForwardにはResidual構造がある。
### 実装の参考
- [pytorch/transformer.py at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py)


## Vision Transformer(ViT)

### 論文概要

#### どんなもの？
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)  

2020年にGoogle Brainが発表したモデル．Transformerから着想を得ている．

#### 先行研究と比べてどこがすごい？
当時，Attentionを画像認識モデルに使用する際にはCNNと組み合わせる，もしくは全体のアーキテクチャを維持しつつ特定のコンポーネントをAttentionで置き換えるという方法があった．しかし，Vision TransformerはCNNから完全に脱却し，Transformerのアーキテクチャをほぼそのまま利用した．大規模なデータセットでの事前学習をすることで，色々なベンチマークにてSOTAを達成した．

#### 技術や手法のキモはどこ？
- Transformer Encoder
  - Vaswani et al.2017のTransformerとは少し違い，Multi Head AttentionやMLP(FeedForward)の前にNormalizationレイヤがある．
  - Multi Head Attentionについては，Vaswani et al.2017のTransformerのものを使用している．

![](./images/vit-encoder.png)

<p style="text-align: center;">画像は<a href="https://arxiv.org/pdf/2010.11929.pdf" target="blank_">論文</a>より引用</p>

#### どうやって有効だと検証した？
JFT-300Mという約3億の画像からなるデータセットで事前学習したViTをImageNet(Real)，CIFAR-10，VTABなどのデータセットでfine tuingすると，当時のSOTAだったBiT-LやNoisy Studentといったモデルを上回るaccuracyを得た．

#### 議論はある？
- 事前学習に使うデータセットが小さいと，既存のSOTAモデルを上回ることができない．ViTは巨大なデータセットによる事前学習で真価を発揮し，さらにデータセットのスケーリングによって性能が改善される余地がある．
- パッチごとの位置エンコーディングは，元の画像に戻した時の距離が近いパッチや列や行が同じパッチでは類似どが高いことがわかった．このように，ViTは切り出されたパッチが元の画像でどこの位置にあったかを学習する．

#### 次に読むべき論文は？

### アーキテクチャ詳細
全体のアーキテクチャは次のようになっている．  
![](./images/vit.png)

<p style="text-align: center;">画像は<a href="https://arxiv.org/pdf/2010.11929.pdf" target="blank_">論文</a>より引用</p>

1. 画像をパッチに分割
    - 元の画像のテンソルが<img src="https://latex.codecogs.com/svg.image?H\times{W\times{C}" title="H\times{W\times{C}" />次元(それぞれ縦，横，チャネル数を表す)のであるとする．これを<img src="https://latex.codecogs.com/svg.image?N" title="N" />個のパッチにするとき，この画像テンソルを
2. パッチのembedding
3. パッチのpositional encodingをパッチのembeddingに足す
4. TransformerのEncoderに複数回パッチのベクトルを通す
5. MLP(多層パーセプトロン)でクラスを分類する

### 実装の参考
- [vit-pytorch/vit.py at main · lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)
