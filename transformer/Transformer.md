

# SubLayers
## Attention
### Scaled Dot Product Attention
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758204626341-90309994-a415-419f-8c0b-904da8b6fe09.png)

$ \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
 $

+ $ Q $**（Query）**: 用于查询的向量矩阵。
+ $ K $**（Key）**: 表示键的向量矩阵，用于与查询匹配。
+ $ V $**（Value）**: 值矩阵，注意力权重最终会作用在该矩阵上。
+ $ d_k $: 键或查询向量的维度。

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch_size, num_heads, seq_len_q, embed_size)
    K: (batch_size, num_heads, seq_len_k, embed_size)
    V: (batch_size, num_heads, seq_len_v, embed_size)

    embed_size: The dimention of input Embedding (d_model).
    """
    embed_size = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1))   # dot product

    scaled_scores = scores / math.sqrt(embed_size)

    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scaled_scores, dim=-1)

    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

### Single-Head Attention
```python
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_size):
        super(Attention, self).__init__()
        self.embed_size = embed_size

        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, mask=None):
        """
        Parameters:
            q: (batch_size, seq_len_q, embed_size)
            k: (batch_size, seq_len_k, embed_size)
            v: (batch_size, seq_len_v, embed_size)
            mask: (batch_size, seq_len_q, seq_len_k)

        Return:
            out: output after attention mechanism
            attention_weights: weighted attention matrix weights after softmax
        """


        # Transform q, k, v using linear layers to Q, K, V
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        out, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        return out, attention_weights
```

### Self-Attention
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758204676540-bda3c7ec-a9ba-4477-93ea-90911af3282f.png)

$ Q = XW^Q, \quad K = XW^K, \quad V = XW^V $

```python
import torch
import torch.nn as nn

class SelfAttetion(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.attention = Attention(embed_size)

    def forward(self, x, mask=None):
        # q = k = v = x
        out, attention_weights = self.attention(x, x, x, mask)

        return out, attention_weights
```

### Cross-Attention
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758204676080-873dcd80-576c-4ebb-a4a2-490a0d09e21a.png)

![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758204710421-1ca076ef-f816-4c06-bff2-4a6033de1672.png)

$ Q = X_{\text{decoder}} W^Q, \quad K = X_{\text{encoder}} W^K, \quad V = X_{\text{encoder}} W^V $

```python
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_size):
        super(CrossAttention, self).__init__()
        self.attention = Attention(embed_size)

    def forward(self, q, kv, mask=None):
        # q from decoder, k, v from encoder
        out, attention_weights = self.attention(q, kv, kv, mask)

        return out, attention_weights
```

#### Q: 现在所说的性能“提升”真的是由多头造成的吗？
不一定。如果**每个头都独立使用线性层且维度等于 **`embed_size`，模型的参数量会比单头模型大很多，此时性能提升可能是因为**参数量的增加**。为了更准确地评估多头机制的实际贡献，我们可以使用以下两种方法进行公平的对比：

1. **方法 1：单头模型增加参数量（与多头模型参数量一致）**使用**一个头**，但将其参数矩阵 $ W^Q, W^K, W^V $ 扩展为：$ `
W \in \mathbb{R}^{d_{\text{model}} \times (d_{\text{model}} \cdot h)}
` $在这种情况下，虽然还是单头模型，但增加了参数量，参数规模将与多头模型保持一致，可以评估性能提升是否真的来自于多头机制本身。
2. **方法 2：降低每个头的维度（与单头模型参数量一致）**降低**每**个头的维度，使得：$ `
h \times \text{head\_dim} = \text{embed\_size}
` $也就是说，每个头的线性变换矩阵 $ W_i^Q, W_i^K, W_i^V $ 的尺寸应为：$ `
W_i \in \mathbb{R}^{d_{\text{model}} \times \text{head\_dim}}
` $其中：$ `
\text{head\_dim} = \frac{\text{embed\_size}}{h}
` $在这种情况下，多头模型的参数规模与单头模型保持一致。

### [Code] Multi-Head Attention
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758204737867-5d926b6d-8e11-4885-a98a-29fdbbffd0ae.png)

+ `embed_size`** → **$ d_{\text{model}} $：输入序列的嵌入维度，即 Transformer 中每个位置的特征向量维度。
+ `num_heads`** → **$ h $：注意力头的数量，即将输入序列拆分为多少个并行的注意力头。
+ `head_dim`** → **$ d_k $：每个注意力头的维度，由 $ d_k = \frac{d_{\text{model}}}{h} $ 计算得到，确保所有头的总维度与嵌入维度一致。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        """
        Parameters:
            d_model: The dimention of input Embedding (d_model).
            h: The number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_model = d_model
        self.h = h

        self.w_q = nn.Linear(d_model, d_model)  # (batch_size, seq_len_q, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)   # (batch_size, seq_len_q, d_model)

    def forward(self, q, k, v, mask=None):
        """
        Parameters:
            q: (batch_size, seq_len_q, d_model)
            k: (batch_size, seq_len_k, d_model)
            v: (batch_size, seq_len_v, d_model)
            mask: (batch_size, 1, seq_len_q, seq_len_k) or (batch_size, h, seq_len_q, seq_len_k)
        
        Return:
            out: output after multi-head attention mechanism
            attention_weights: weighted attention matrix weights after softmax for each head
        """
    batch_size = q.shape[0]

        seq_len_q = q.size(1)
        seq_len_k = k.size(1)

        Q = self.w_q(q).view(batch_size, seq_len_q, self.h, -1).transpose(1, 2)   # (batch_size, h, seq_len_q, d_model/h)
        K = self.w_k(k).view(batch_size, seq_len_k, self.h, -1).transpose(1, 2)
        V = self.w_v(v).view(batch_size, seq_len_k, self.h, -1).transpose(1, 2)

        scaled_attention, _ = scaled_dot_product_attention(Q, K, V, mask)   # (batch_size, h, seq_len_q, d_model/h)

        concat_out = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len_q, d_model)

        out = self.fc_out(concat_out)   # (batch_size, seq_len_q, d_model)

        return out
    
    def scaled_dot_product_attention(Q, K, V, mask):
        """
        Parameters:
            Q: (batch_size, h, seq_len_q, d_model/h)
            K: (batch_size, h, seq_len_k, d_model/h)
            V: (batch_size, h, seq_len_v, d_model/h)
            mask: (1, 1, seq_len_q, seq_len_k) or (batch_size, 1, seq_len_q, seq_len_k) or (batch_size, h, seq_len_q, seq_len_k)
        Return:
            out: (batch_size, h, seq_len_q, d_model/h)
            attention_weights: (batch_size, h, seq_len_q, seq_len_k)
        """
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (batch_size, h, seq_len_q, seq_len_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)   # (batch_size, h, seq_len_q, seq_len_k)

        output = torch.matmul(attention_weights, V)     # (batch_size, h, seq_len_q, d_k)

        return output, attention_weights
```

```python
mask matrix:
tensor([[[[1., 0., 0.],
          [1., 1., 0.],
          [1., 1., 1.]]]])

weighted attention matrix:
tensor([[[[1.0000, 0.0000, 0.0000],
          [0.4543, 0.5457, 0.0000],
          [0.0393, 0.2787, 0.6820]],

         [[1.0000, 0.0000, 0.0000],
          [0.6347, 0.3653, 0.0000],
          [0.3787, 0.2274, 0.3938]]],


        [[[1.0000, 0.0000, 0.0000],
          [0.2770, 0.7230, 0.0000],
          [0.2145, 0.3288, 0.4567]],

         [[1.0000, 0.0000, 0.0000],
          [0.9932, 0.0068, 0.0000],
          [0.1146, 0.4249, 0.4605]]]])
```

## Feed Forward
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758204756502-ffa35c9a-2c28-449e-8eca-f6ca767cc8e8.png)

$ \text{FFN}(x_i) = \text{max}(0, x_i W_1 + b_1) W_2 + b_2 $

其中：

+ $ x_i \in \mathbb{R}^{d_{\text{model}}} $ 表示第 $ i $ 个位置的输入向量。 
+ $ W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}} $ 和 $ W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}} $ 是两个线性变换的权重矩阵。
+ $ b_1 \in \mathbb{R}^{d_{\text{ff}}} $ 和 $ b_2 \in \mathbb{R}^{d_{\text{model}}} $ 是对应的偏置向量。
+ $ \text{max}(0, \cdot) $ 是 **ReLU 激活函数**，用于引入非线性。

### [Code] PositionwiseFeedForward
```python
import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Parameters:
            d_model: The dimention of input Embedding (d_model).
            d_ff: The dimension of the hidden layer in the feed-forward network.
            dropout: Dropout rate.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # first linear layer + ReLU activation
        # then second linear layer
        return self.w_2(self.w_1(x).relu()) # self.w_2(self.dropout(self.w_1(x).relu()))
```

:::warning
所以 FFN 本质就是两个线性变换之间嵌入了一个 **ReLU** 激活函数，实现起来非常简单。

:::

## Residual Connection & Layer Norm
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758204774013-081e7abb-4c96-4d99-891a-3ae40e435e9b.png)

### Residual Connection
残差连接是一种跳跃连接（Skip Connection），它将层的输入直接加到输出上（观察架构图中的箭头），对应的公式如下：

$ \text{Output} = \text{SubLayer}(x) + x $

这种连接方式有效缓解了**深层神经网络的梯度消失**问题。

```python
import torch
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self, dropout=0.1):
        """
        Residual Connection: add residual aand dropout after each sublayers

        Parameters:
            dropout: Dropout possibility, apply to sublayer output before residual connection, prevent overfit
        """
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
        Parameters:
            x: input tensor (batch_size, seq_len, d_model)
            sublayer: a function that takes x as input and returns output of the sublayer

        Return:
            output after applying sublayer, dropout and adding residual connection (batch_size, seq_len, d_model)
        """
        return x + self.dropout(sublayer(x))
```

#### Q: 为什么可以缓解梯度消失？
首先，我们需要了解什么是梯度消失。

在深度神经网络中，参数的梯度通过反向传播计算，其公式为：

$ \frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial h_n} \cdot \frac{\partial h_n}{\partial h_{n-1}} \cdot \ldots \cdot \frac{\partial h_1}{\partial W} $

当网络层数增加时，**链式法则**中的梯度相乘可能导致梯度值越来越小（梯度消失）或越来越大（梯度爆炸），使得模型难以训练和收敛。

假设输出层的损失为 $ \mathcal{L} $, 且 $ \text{SubLayer}(x) $ 表示为 $ F(x) $。在没有残差连接的情况下，梯度通过链式法则计算为：

$ \frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial F(x)} \cdot \frac{\partial F(x)}{\partial x} $

如果 $ \frac{\partial F(x)}{\partial x} $ 的绝对值小于 1，那么随着层数的增加，梯度会呈快速缩小，导致梯度消失。

引入残差连接后，输出变为 $ F(x) + x $, 其梯度为：

$ \frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial (x + F(x))} \cdot (1 + \frac{\partial F(x)}{\partial x}) $

这里，包含了一个常数项 1，这意味着即使 $ \frac{\partial F(x)}{\partial x} $ 很小，梯度仍然可以有效地反向传播，缓解梯度消失问题。

### [Code] Layer Norm
假设输入向量为 $ x = (x_1, x_2, \dots, x_d) $, LayerNorm 的计算步骤如下：

1. **计算均值和方差**：  
对输入的所有特征求均值 $ \mu $ 和方差 $ \sigma^2 $：$ \mu = \frac{1}{d} \sum_{j=1}^{d} x_j, \quad 
\sigma^2 = \frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2 $
2. **归一化公式**：  
将输入特征 $ \hat{x}_i $ 进行归一化：$ \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
 $其中, $ \epsilon $ 是一个很小的常数（比如 1e-9），用于防止除以零的情况。
3. **引入可学习参数**：  
归一化后的输出乘以 $ \gamma $ 并加上 $ \beta $, 公式如下：$ \text{Output} = \gamma \hat{x} + \beta $其中 $ \gamma $ 和 $ \beta $ 是可学习的参数，用于进一步调整归一化后的输出。

```python
class LayerNorm(nn.Module):
    def __init__(self, feature_size, epsilon=1e-9):
        """
        Layer Normalization

        Parameters:
            feature_size: The dimension of the input features (d_model).
            epsilon: A small value to avoid division by zero.
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(feature_size))  # Scale parameter, initialized to 1
        self.beta = nn.Parameter(torch.zeros(feature_size)) # Shift parameter, initialized to 0
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta
```

#### Q: BatchNorm 和 LayerNorm 的区别
如果你听说过 **Batch Normalization (BatchNorm)**，或许会疑惑于二者的区别。

假设输入张量的形状为 **(batch_size, feature_size)**，其中 `batch_size=32`，`feature_size=512`。

+ **batch_size**：表示批次中的样本数量。  
+ **feature_size**：表示每个样本的特征维度，即每个样本包含 512 个特征。

这里的一行对应于一个样本，一列对应于一种特征属性。

+ BatchNorm 基于一个**批次**（batch）内的所有样本，针对**特征维度**（列）进行归一化，即在每一列（相同特征或嵌入维度上的 batch_size 个样本）上计算均值和方差。
    - 对第 $ j $ 列（特征）计算均值和方差：$ `
\mu_j = \frac{1}{\text{batch\_size}} \sum_{i=1}^{\text{batch\_size}} x_{i,j}, \quad 
\sigma^2_j = \frac{1}{\text{batch\_size}} \sum_{i=1}^{\text{batch\_size}} (x_{i,j} - \mu_j)^2
` $
+ LayerNorm 基于**每个样本的所有特征**，针对**样本自身**（行内所有特征）进行归一化，即在每一行（一个样本的 embed_size 个特征）上计算均值和方差。
    - 对第 $ i $ 行（样本）计算均值和方差：$ `
\mu_i = \frac{1}{\text{feature\_size}} \sum_{j=1}^{\text{feature\_size}} x_{i,j}, \quad 
\sigma^2_i = \frac{1}{\text{feature\_size}} \sum_{j=1}^{\text{feature\_size}} (x_{i,j} - \mu_i)^2
` $

用表格说明：

| 操作 | 处理维度 | 解释 |
| --- | --- | --- |
| **BatchNorm** | 对列（特征维度）归一化 | 每个特征在所有样本中的归一化 |
| **LayerNorm** | 对行（样本内的特征维度）归一化 | 每个样本的所有特征一起归一化 |


> BatchNorm 和 LayerNorm 在视频中也有讲解：[Transformer论文逐段精读【论文精读】25:40 - 32:04 部分](https://www.bilibili.com/video/BV1pu411o7BE/?share_source=copy_web&vd_source=e46571d631061853c8f9eead71bdb390&t=1540)，不过需要注意的是在 26:25 处应该除以的是标准差而非方差。
>
> ![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758187688747-528c80d0-c998-48f8-82ec-6845f77a8ec8.png)
>
> 对于三维张量，比如图示的 (batch_size, seq_len, feature_size)，可以从立方体的左侧(batch_size, feature_size) 去看成二维张量进行切片。
>

### [Code] SublayerConnection
```python
class SublayerConnection(nn.Module):
    def __init__(self, feature_size, dropout=0.1, epsilon=1e-9):
        """
        Sublayer Connection with Layer Normalization and Residual Connection
        
        Parameters:
            feature_size: The dimension of the input features (d_model).
            dropout: Dropout possibility, apply to sublayer output before residual connection, prevent overfit
            epsilon: A small value to avoid division by zero in LayerNorm
        """
        super(SublayerConnection, self).__init__()
        self.residual = ResidualConnection(dropout)
        self.norm = LayerNorm(feature_size, epsilon)

    def forward(self, x, sublayer):
        # Apply dropout to sublayer output, then residual connection, then Layer Normalization
        return self.norm(self.residual(x, sublayer))
```

这里是 Post-Norm，即残差连接后进行 LayerNorm，和 Transformer 论文的表述一致。另一种实现是 Pre-Norm，即在进入子层计算之前先进行 LayerNorm： ` return x + self.dropout(sublayer(self.norm(x)))`。

```python
class SublayerConnections(nn.Module):
    def __init__(self, feature_size, dropout=0.1, epsilon=1e-9):
        """
        Stack of Sublayer Connections

        Parameters:
            feature_size: The dimension of the input features (d_model).
            dropout: Dropout possibility, apply to sublayer output before residual connection, prevent overfit
            epsilon: A small value to avoid division by zero in LayerNorm
        """
        super(SublayerConnections, self).__init__()
        self.norm = LayerNorm(feature_size, epsilon)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # Apply dropout to sublayer output, then residual connection, then Layer Normalization
        return self.norm(x + self.dropout(sublayer(x)))
```

## Embeddings
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758204791336-6dc7cf1f-97a3-496d-b1e5-1107c7feae7c.png)

模型实际操作的是**tensor**，而非**string**。在将输入文本传递给模型之前，首先需要进行**tokenization**，即将文本拆解为多个 t**oken**，随后这些 token 会被映射为对应的 **token ID**，从而转换为模型可理解的数值形式。

此时，数据的形状为 `(seq_len,)`，其中 `seq_len` 表示输入序列的长度。

+ `nn.Embedding`：创建嵌入层，将词汇表中的每个 token ID 映射为对应的嵌入向量。
+ `vocab_size`：词汇表的大小。
+ `d_model`：嵌入向量的维度大小。
+ **缩放嵌入（Scaled Embedding）**：将嵌入层的输出（参数）乘以 $ \sqrt{d_{\text{model}}} $。

```python
import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        Parameters:
            vocab_size: The size of the vocabulary.
            d_model: The dimension of the embedding (d_model).
        """
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.scale_factor = math.sqrt(d_model)

    def forward(self, x):
        """
        Parameters:
            x: input tensor of shape (batch_size, seq_len) containing token IDs.

        Return:
            output: embedded tensor of shape (batch_size, seq_len, d_model)
        """
        return self.embed(x) * self.scale_factor
```

#### Q: 为什么需要嵌入层？
因为 token ID 只是整数标识符，彼此之间没有内在联系。如果直接使用这些整数，模型可能在训练过程中学习到一些模式，但无法充分捕捉词汇之间的语义关系，这显然不足以支撑起现在的大模型。

举个简单的例子来理解“语义”关系：像“猫”和“狗”在向量空间中的表示应该非常接近，因为它们都是宠物；“男人”和“女人”之间的向量差异可能代表性别的区别。此外，不同语言的词汇，如“男人”（中文）和“man”（英文），如果在相同的嵌入空间中，它们的向量也会非常接近，反映出跨语言的语义相似性。同时，【“女人”和“woman”（中文-英文）】与【“男人”和“man”（中文-英文）】之间的差异也可能非常相似。

对于模型而言，没有语义信息就像我们小时候刚开始读英语阅读报：“这些字母拼起来是什么？不知道。这些单词在说什么？不知道。”囫囵吞枣看完后去做题：“嗯，昨天对答案的时候，A 好像多一点，其他的差不多，那多选一点 A，其他平均分 :)。”

所以，为了让模型捕捉到 token 背后复杂的语义（Semantic meaning）关系，我们需要将离散的 token ID 映射到一个高维的连续向量空间（Continuous, dense）。这意味着每个 token ID 会被转换为一个**嵌入向量**（embedding vector），期望通过这种方式让语义相近的词汇在向量空间中距离更近，使模型能更好地捕捉词汇之间的关系。当然，简单的映射无法做到这一点，因此需要“炼丹”——是的，嵌入层是可以训练的。

#### Q: 什么是 nn.Embedding()？和 nn.Linear() 的区别是什么？
其实非常简单，`nn.Embedding()` 就是从权重矩阵中查找与输入索引对应的行，类似于查找表操作，而 `nn.Linear()` 进行线性变换。直接对比二者的 `forward()` 方法：

```python
# Embedding
def forward(self, input):
    return self.weight[input]  # 没错，就是返回对应的行

# Linear
def forward(self, input):
    torch.matmul(input, self.weight.T) + self.bias
```

运行下面的代码来验证：

```python
import torch
import torch.nn as nn

# 设置随机种子
torch.manual_seed(42)

# nn.Embedding() 权重矩阵形状为 (num_embeddings, embedding_dim)
num_embeddings = 5  # 假设有 5 个 token
embedding_dim = 3   # 每个 token 对应 3 维 embeddings

# 初始化嵌入层
embedding = nn.Embedding(5, 3)

# 整数索引
input_indices = torch.tensor([0, 2, 4])

# 查找嵌入
output = embedding(input_indices)

# 打印结果
print("权重矩阵：")
print(embedding.weight.data)
print("\nEmbedding 输出：")
print(output)
```

```sql
权重矩阵：
tensor([[ 0.3367,  0.1288,  0.2345],
  [ 0.2303, -1.1229, -0.1863],
  [ 2.2082, -0.6380,  0.4617],
  [ 0.2674,  0.5349,  0.8094],
  [ 1.1103, -1.6898, -0.9890]])

Embedding 输出：
tensor([[ 0.3367,  0.1288,  0.2345],
  [ 2.2082, -0.6380,  0.4617],
  [ 1.1103, -1.6898, -0.9890]], grad_fn=<EmbeddingBackward0>)
```

**要点**：

+ **权重矩阵**：嵌入层的权重矩阵，其形状为 `(num_embeddings, embedding_dim)`，熟悉线性层的同学可以理解为 `(in_features, out_features)`。
+ **Embedding 输出**：根据输入indices，从权重矩阵中提取对应的embeddings向量（行）。
+ 在例子中，输入indices `[0, 2, 4]`，因此输出了权重矩阵中第 0、2、4 行对应的嵌入向量。

## Softmax
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758204793643-1d0fa146-64ec-4626-aaa0-4e7c43bc32c7.png)

**Softmax** 函数是一种常用的激活函数，能够将任意实数向量转换为**概率分布**，确保每个元素的取值范围在 [0, 1] 之间，并且所有元素的和为 1。其数学定义如下：

$ \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}} $

其中：

+ $ x_i $ 表示输入向量中的第 $ i $ 个元素。
+ $ \text{Softmax}(x_i) $ 表示输入 $ x_i $ 转换后的概率。

:::warning
我们可以把 Softmax 看作一种**归一化的指数变换**。相比于简单的比例归一化 $ \frac{x_i}{\sum_j x_j} $, Softmax 通过指数变换放大数值间的差异，让较大的值对应更高的概率，同时避免了负值和数值过小的问题。

:::

```python
import torch
import torch.nn as nn

def softmax(x):
    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x, dim=-1, keepdim=True)
    return exp_x / sum_exp_x

x = torch.tensor([1.0, 2.0, 3.0])

result = softmax(x)

softmax = nn.Softmax(dim=-1)
nn_result = softmax(x)

print("Custom softmax result:", result)
print("nn.Softmax result:", nn_result)
```

```sql
Custom softmax result: tensor([0.0900, 0.2447, 0.6652])
nn.Softmax result: tensor([0.0900, 0.2447, 0.6652])
```

### Cross-Entropy Loss
在多分类任务中，Softmax 通常与**交叉熵损失**（Cross-Entropy Loss）配合使用（Transformer 同样如此）。交叉熵损失用于衡量模型预测的概率分布与真实分布之间的差异，其数学公式如下：

$ \mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i) $

其中：

+ $ y_i $ 是真实标签（ground-truth）的 one-hot 编码。
+ $ \hat{y}_i $ 是模型的预测概率（即 Softmax 的输出）。

#### Q: Transformer 模型的输出是概率还是 logits？
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758196634090-3af7b735-41f9-40d4-b71e-fdaae50fb67c.png)

+ **常规做法**：输出 logits + `CrossEntropyLoss()`。
    - 输出：**logits**（最后一层线性层的原始结果，没过 softmax）
    - 损失：`nn.CrossEntropyLoss()` 它内部会自动做 softmax + log，不需要你自己处理。
+ **另一种做法**：输出概率 + `log_softmax()` + `NLLLoss()`。
    - 因为 `NLLLoss` 接收的是 **log(概率)**，所以需要在 softmax 后再 `torch.log()`。

```python
logits = torch.tensor([[2.0, 0.5], [0.5, 2.0]])  # 未经过 softmax 的 logits
target = torch.tensor([0, 1])  # 目标标签

# 使用 nn.CrossEntropyLoss 计算损失（接受 logits）
criterion_ce = nn.CrossEntropyLoss()
loss_ce = criterion_ce(logits, target)

# 使用 softmax 后再使用 nn.NLLLoss 计算损失
log_probs = F.log_softmax(logits, dim=1)
criterion_nll = nn.NLLLoss()
loss_nll = criterion_nll(log_probs, target)

print(f"Loss using nn.CrossEntropyLoss: {loss_ce.item()}")
print(f"Loss using softmax + nn.NLLLoss: {loss_nll.item()}")
```

```sql
Loss using nn.CrossEntropyLoss: 0.2014133334159851
Loss using softmax + nn.NLLLoss: 0.2014133334159851
```

## Positional Encoding
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758204819421-391fe3eb-7872-4ac7-9557-a603caf28c58.png)

所以如果嵌入向量本身不包含位置信息，就意味着**输入元素的顺序不会影响输出的权重计算，模型无法从中捕捉到序列的顺序信息**

为了解决这个问题，Transformer 引入了**Positional Encoding**：为每个位置生成一个向量，这个向量与对应的Embedding向量相加，从而在输入中嵌入位置信息。

$ PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \\
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) $

其中：

+ $ pos $ 表示位置索引（Position）
+ $ i $ 表示维度索引
+ $ d_{\text{model}} $ 是嵌入向量的维度
+ $ div\_term[i] = 10000^{-\frac{2i}{d_{\text{model}}}} $
    - **低维度 (i 小)** → 分母接近 1，频率高 → sin/cos 摆动很快。
    - **高维度 (i 大)** → 分母很大，频率低 → sin/cos 摆动很慢。

这样一来，每个位置会被编码成一组 **不同频率的波形组合**。模型看到这些波形，就能区分位置差异。

```python
import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Positional Encoding for Transformer models

        Parameters:
            d_model: The dimension of the embedding (d_model).
            dropout: Dropout rate.
            max_len: Maximum length of the input sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)    # apply dropout to the sum of input embeddings and positional encodings

        # Positional encoding matrix (max_len, d_model)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # Position indices (max_len, 1)

        # Compute the positional encodings for each dimension
        # torch.arange(0, d_model, 2) = 2i
        # (-math.log(10000.0) / d_model) = -ln(10000) / d_model
        # div_term = exp(2i * -ln(10000) / d_model) = 100000^(-2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Combine position and frequency to compute sin and cos
        pe[:, 0::2] = torch.sin(position * div_term)    # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)    # odd dimensions

        # Add a batch dimension, so that pe can be added to input embeddings, shape (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register pe as a buffer, not a parameter, so it won't be updated during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Parameters:
            x: input tensor of shape (batch_size, seq_len, d_model)

        Return:
            output: tensor of shape (batch_size, seq_len, d_model) after adding positional encodings and applying dropout
        """
        # Get the positional encodings for the input sequence length
        x = x + self.pe[:, :x.size(1), :]  # Add positional encodings to input embeddings
        return self.dropout(x)
```

## Input Processing
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758199802362-784b969f-05ce-41eb-87e9-dbeb612f2d38.png)

### [Code] Encoder Inputs
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758199804878-11918256-704a-4ce0-b77b-e0c407189009.png)

Input Embedding + Positional Encoding -> (Source Embedding)

```python
class SourceEmbedding(nn.Module):
    def __init__(self, src_vocab_size, d_model, dropout=0.1):
        """
        Source Embedding: combines token embeddings and positional encodings for the source sequence

        Parameters:
            src_vocab_size: The size of the source vocabulary.
            d_model: The dimension of the embedding (d_model).
            dropout: Dropout rate.
        """
        super(SourceEmbedding, self).__init__()
        self.embed = Embeddings(src_vocab_size, d_model)    # token embeddings
        self.positional_encoding = PositionalEncoding(d_model, dropout) # positional encodings

    def forward(self, x):
        """
        Parameters:
            x: input tensor of shape (batch_size, seq_len_src) containing token IDs.

        Return:
            output: tensor of shape (batch_size, seq_len_src, d_model) after adding token embeddings and positional encodings
        """
        x = self.embed(x)  # word embeddings (batch_size, seq_len_src, d_model)
        x = self.positional_encoding(x)  # positional encodings (batch_size, seq_len_src, d_model)
        return x

```

### [Code] Decoder Inputs
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758199819446-2a899fd9-816f-4294-87af-5b58be119172.png)

Input Embedding + Positional Encoding ->Target Embedding

```python
class TargetEmbedding(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, dropout=0.1):
        """
        Target Embedding: combines token embeddings and positional encodings for the target sequence

        Parameters:
            tgt_vocab_size: The size of the target vocabulary.
            d_model: The dimension of the embedding (d_model).
            dropout: Dropout rate.
        """
        super(TargetEmbedding, self).__init__()
        self.embed = Embeddings(tgt_vocab_size, d_model)    # token embeddings
        self.positional_encoding = PositionalEncoding(d_model, dropout) # positional encodings

    def forward(self, x):
        """
        Parameters:
            x: input tensor of shape (batch_size, seq_len_tgt) containing token IDs.

        Return:
            output: tensor of shape (batch_size, seq_len_tgt, d_model) after adding token embeddings and positional encodings
        """
        x = self.embed(x)  # word embeddings (batch_size, seq_len_tgt, d_model)
        x = self.positional_encoding(x)  # positional encodings (batch_size, seq_len_tgt, d_model)
        return x
```

#### Q: 什么是右移（shifted right）？
目标输出序列 $ Y = (y_1, y_2, ..., y_T) $ 向右移动一位，生成一个新的序列 $ Y' = (0, y_1, y_2, ..., y_{T-1}) $，其中第一个位置用填充标记（假设为 0 或特定的开始标记 `<sos>`）占位。

需要注意的是，这个操作位于嵌入（Embedding）之前，可以将公式 $ Y $ 中的元素当作 token id。



假设目标序列是 “I love NLP”，在右移之后得到输入序列：

```python
<sos> I love NLP
```

这样在训练时就能避免“偷看”当前位置，具体预测过程如下：

+ 输入 `<sos>` 预测 “I”
+ 输入 `<sos> I` 预测 “love”
+ 输入 `<sos> I love` 预测 “NLP”

这是一个非常优雅的解决方案，与掩码协同工作，防止模型在训练时泄露未来信息。



另外，右移操作通常不是在 Transformer 模型内部处理的，而是在数据传入模型之前的预处理阶段。例如，假设目标序列已经经过分词并添加了特殊标记：

```python
tgt = "<sos> I love NLP <eos>"
```

我们需要从这个序列获取输入和输出：

```python
tgt_input = tgt[:-1]  # "<sos> I love NLP"
tgt_output = tgt[1:]  # "I love NLP <eos>"
```

在训练过程中，解码器将使用 `tgt_input` 来预测 `tgt_output`。具体过程如下：

+ 输入 `<sos>` 预测 “I”
+ 输入 `<sos> I` 预测 “love”
+ 输入 `<sos> I love` 预测 “NLP”
+ 输入 `<sos> I love NLP` 预测 `<eos>`

> [!note]
>
> 当前的 `tgt` 是一维的字符串。如果要处理一个**batch**中的数据，需要对张量的每一维进行切片。例如，对于形状为 `(batch_size, seq_len)` 的张量 `tgt`：
>
> 这样，`tgt_input` 和 `tgt_output` 分别对应批次序列的输入和目标输出，用于模型训练。
>

```python
tgt_input = tgt[:, :-1]  # 去除每个序列的最后一个token
tgt_output = tgt[:, 1:]  # 去除每个序列的第一个token
```

## Mask
mask用于控制注意力机制中哪些位置需要被忽略

#### Q: 为什么需要 Mask 机制？
+ **填充掩码（Padding Mask）**在处理不等长的输入序列时，需要使用填充符（padding）补齐短序列。在计算注意力时，填充部分不应对结果产生影响（q 与填充部分的 k 匹配程度应该为 0），因此需要使用填充掩码忽略这些位置。
+ **未来掩码（Look-ahead Mask）**在训练自回归模型（如 Transformer 中的解码器）时，为了防止模型“偷看”未来的词，需要用掩码屏蔽未来的位置，确保模型只能利用已知的上下文进行预测。

### [Code] Padding Mask
在注意力计算时屏蔽填充 `<PAD>` 位置，防止模型计算注意力权重的时候考虑这些无意义的位置，在编码器的自注意力中使用。

```python
def create_padding_mask(seq, pad_token_id=0):
    # seq: (batch_size, seq_len)
    mask = (seq != pad_token_id).unsqueeze(1).unsqueeze(2)    # (batch_size, 1, 1, seq_len)
    return mask # 1 for non-pad tokens, 0 for pad tokens

seq = torch.tensor([[5, 7, 9, 0, 0], [8, 6, 0, 0, 0]])  # 0 is <PAD>
print(create_padding_mask(seq))
```

```python
tensor([[[[ True,  True,  True, False, False]]],
        [[[ True,  True, False, False, False]]]])
```

### Look-ahead Mask
用于在解码器中屏蔽未来的位置，防止模型在预测下一个词时“偷看”答案（训练时），在解码器中使用。

```python
def create_look_ahead_mask(size):
    mask = torch.tril(torch.ones(size, size)).type(torch.bool)  # (size, size)
    return mask  # (seq_len, seq_len)

print(create_look_ahead_mask(5))
```

```python
tensor([[ True, False, False, False, False],
        [ True,  True, False, False, False],
        [ True,  True,  True, False, False],
        [ True,  True,  True,  True, False],
        [ True,  True,  True,  True,  True]])
```

在第`i`个位置，模型只能看到位置 `0`到 `i`，而屏蔽位置 `i+1` 及之后的信息。

### [Code] Combined Mask
```python
def create_decoder_mask(tgt_seq, pad_token_id=0):
    padding_mask = create_padding_mask(tgt_seq, pad_token_id)  # (batch_size, 1, 1, seq_len_tgt)
    look_ahead_mask = create_look_ahead_mask(tgt_seq.size(1)).to(tgt_seq.device)  # (seq_len_tgt, seq_len_tgt)

    combined_mask = look_ahead_mask.unsqueeze(0) & padding_mask  # (batch_size, 1, seq_len_tgt, seq_len_tgt)
    return combined_mask  # (batch_size, 1, seq_len_tgt, seq

tgt_seq = torch.tensor([[1, 2, 3, 0, 0]])  # 0 is <PAD>
print(create_decoder_mask(tgt_seq))
```

```python
tensor([[[[ True, False, False, False, False],
          [ True,  True, False, False, False],
          [ True,  True,  True, False, False],
          [ True,  True,  True,  True, False],
          [ True,  True,  True,  True, False]]]])
```

# Construct Model
## [Code] Encoder Layer
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758201603921-266cf42e-7c3f-4639-83f8-432c098f53f3.png)

**组件**：

+ Multi-Head Self-Attention
+ Feed Forward
+ Add & Norm（SublayerConnection）

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        """
        Encoder Layer

        Parameters:
            d_model: The dimension of the embedding (d_model).
            h: The number of heads.
            d_ff: The dimension of the hidden layer in the feed-forward network.
            dropout: Dropout rate.
        """
    super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Define two sublayer connections for self-attention and feed-forward network (as two residual connections in model)
        self.sublayers = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, src_mask):
        """
        Feed forward Function
        
        Parameters:
            x: input tensor (batch_size, seq_len_src, d_model)
            src_mask: source mask for self-attention (batch_size, 1, 1, seq_len_src)
        
        Return:
            output tensor of shape (batch_size, seq_len_src, d_model)
        """
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, src_mask))   # Self-attention sublayer
        x = self.sublayers[1](x, self.feed_forward)  # Feed-forward sublayer
        return x

```

## [Code] Decoder Layer
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758202953465-bda41818-87b7-4c2e-85f1-808e843b6f91.png)

**组件**：

+ Masked Multi-Head Self-Attention
+ Multi-Head Cross-Attention
+ Feed Forward
+ Add & Norm（SublayerConnection）

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        """
        Decoder Layer

        Parameters:
            d_model: The dimension of the embedding (d_model).
            h: The number of heads.
            d_ff: The dimension of the hidden layer in the feed-forward network.
            dropout: Dropout rate.
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h) # Masked Multi-Head Self-Attention
        self.cross_attn = MultiHeadAttention(d_model, h) # Multi-Head Cross-Attention
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout) # Feed-Forward Network

        # Define three sublayer connections for self-attention, cross-attention and feed-forward network
        self.sublayers = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])
        self.d_model = d_model

    def forward(self, x, enc_output, tgt_mask, src_mask):
        """
        Feed forward Function
        
        Parameters:
            x: input tensor (batch_size, seq_len_tgt, d_model)
            enc_output: encoder output tensor (batch_size, seq_len_src, d_model)
            tgt_mask: target mask for self-attention (batch_size, 1, seq_len_tgt, seq_len_tgt)
            src_mask: source mask for cross-attention (batch_size, 1, 1, seq_len_src)
        
        Return:
            output tensor of shape (batch_size, seq_len_tgt, d_model)
        """
        # First sublayer: masked multi-head self-attention
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # Second sublayer: multi-head cross-attention with encoder output
        x = self.sublayers[1](x, lambda x: self.cross_attn(x, enc_output, enc_output, src_mask))

        # Third sublayer: position-wise feed-forward network
        x = self.sublayers[2](x, self.feed_forward)
        
        return x
```

## [Code] Encoder
```python
class Encoder(nn.Module):
    def __init__(self, d_model, N, h, d_ff, dropout=0.1):
        """
        Encoder: stack of N encoder layers

        Parameters:
            d_model: The dimension of the embedding (d_model).
            N: The number of encoder layers.
            h: The number of heads.
            d_ff: The dimension of the hidden layer in the feed-forward network.
            dropout: Dropout rate.
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        """
        Feed forward Function
        
        Parameters:
            x: input tensor (batch_size, seq_len, d_model)
            mask: source mask for self-attention (batch_size, 1, 1, seq_len)

        Return:
            output tensor of shape (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) # Final Layer Normalization
```

## [Code] Decoder
```python
class Decoder(nn.Module):
    def __init__(self, d_model, N, h, d_ff, dropout=0.1):
        """
        Decoder: stack of N decoder layers

        Parameters:
            d_model: The dimension of the embedding (d_model).
            N: The number of decoder layers.
            h: The number of heads.
            d_ff: The dimension of the hidden layer in the feed-forward network.
            dropout: Dropout rate.
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self.norm = LayerNorm(d_model)  # Final Layer Normalization

    def forward(self, x, enc_output, tgt_mask, src_mask):
        """
        Feed forward Function

        Parameters:
            x: input tensor (batch_size, seq_len_tgt, d_model)
            enc_output: encoder output tensor (batch_size, seq_len_src, d_model)
            tgt_mask: target mask for self-attention (batch_size, 1, seq_len_tgt, seq_len_tgt)
            src_mask: source mask for cross-attention (batch_size, 1, 1, seq_len_src)

        Return:
            output tensor of shape (batch_size, seq_len_tgt, d_model)
        """
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return self.norm(x) # Final Layer Normalization         
```

## [Code] Model
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758203626865-3b4a67d7-9d39-4fd9-a807-0f595d0b63bf.png)

+ **Input Embedding and Position Encoding:**
    - **SourceEmbedding:** Embeds the source sequence and adds positional encoding.
    - **TargetEmbedding:** Embeds the target sequence and adds positional encoding.
+ **Multi-Head Attention and Feed-Forward Network:**
    - **MultiHeadAttention:** Multi-head attention mechanism.
    - **PositionwiseFeedForward:** Position-wise feed-forward network.
+ **Encoder and Decoder:**
    - **Encoder:** Stacked from multiple `EncoderLayer`s.
    - **Decoder:** Stacked from multiple `DecoderLayer`s.
+ **Output Layer:**
    - **fc_out:** Linear layer that maps the decoder output to the target vocabulary dimension.

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, N, h, d_ff, dropout=0.1):
        """
        Transformer Model

        Parameters:
            src_vocab_size: The size of the source vocabulary.
            tgt_vocab_size: The size of the target vocabulary.
            d_model: The dimension of the embedding (d_model).
            N: The number of encoder and decoder layers.
            h: The number of heads.
            d_ff: The dimension of the hidden layer in the feed-forward network.
            dropout: Dropout rate.
        """
        super(Transformer, self).__init__()

        # Input Embeddings and Positional Encodings, src for encoder, tgt for decoder
        self.encoder_embedding = SourceEmbedding(src_vocab_size, d_model, dropout)
        self.decoder_embedding = TargetEmbedding(tgt_vocab_size, d_model, dropout)

        # Encoder and Decoder stacks
        self.encoder = Encoder(d_model, N, h, d_ff, dropout)
        self.decoder = Decoder(d_model, N, h, d_ff, dropout)

        # Output linear layer to project decoder output to vocabulary size
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)  # Final linear layer to project to vocab size

    def forward(self, src, tgt):
        """
        Feed forward Function

        Parameters:
            src: source input tensor (batch_size, seq_len_src) containing token IDs
            tgt: target input tensor (batch_size, seq_len_tgt) containing token IDs

        Return:
            output tensor of shape (batch_size, seq_len_tgt, tgt_vocab_size)
        """
        # Create masks
        src_mask = create_padding_mask(src)  # (batch_size, 1, 1, seq_len_src)
        tgt_mask = create_decoder_mask(tgt)  # (batch_size, 1, seq_len_tgt, seq_len_tgt)

        # Encoder
        enc_output = self.encoder(self.encoder_embedding(src), src_mask)  # (batch_size, seq_len_src, d_model)

        # Decoder
        dec_output = self.decoder(self.decoder_embedding(tgt), enc_output, tgt_mask, src_mask)  # (batch_size, seq_len_tgt, d_model)

        # Final linear layer to project to vocabulary size
        output = self.fc_out(dec_output)  # (batch_size, seq_len_tgt, tgt_vocab_size)

        return output
```



# QA
### Complexity
| 网络层类型 | 每层计算复杂度 | 顺序操作次数 | 最大路径长度 |
| --- | --- | --- | --- |
| 自注意力（Self-Attention） | $ O(n^2 \cdot d) $ | $ O(1) $ | $ O(1) $ |
| 循环网络（Recurrent Layer） | $ O(n \cdot d^2) $ | $ O(n) $ | $ O(n) $ |
| 卷积网络（Convolution） | $ O(k \cdot n \cdot d^2) $ | $ O(1) $ | $ O(\log_k(n)) $ |
| 受限自注意力（Restricted Self-Attention） | $ O(r \cdot n \cdot d) $ | $ O(1) $ | $ O(n/r) $ |


**符号说明**：

+ $ n $：输入序列的长度（seq_len)，即模型需要处理的序列中包含的元素个数（如文本序列中的词汇数量）。
+ $ d $：表示的维度 (embed_size, d_model)，每个输入元素被映射到的特征空间的维度。
+ $ k $：卷积核（kernal）的大小，在卷积神经网络（CNN）中决定感受野的大小。
+ $ r $：受限注意力机制中“窗口大小”，即每个元素只能与它周围 $ r $ 个邻近元素进行计算，不能算全部的。

#### Q1: 自注意力中每层的计算复杂度怎么计算？
自注意力层的计算复杂度主要来源于以下两个部分：

1. **计算注意力权重**：
    - **查询-键点积计算**：
        * 查询矩阵 $ Q \in \mathbb{R}^{n \times d} $ 与键矩阵 $ K \in \mathbb{R}^{n \times d} $ 的转置 $ K^\top \in \mathbb{R}^{d \times n} $ 进行矩阵乘法，得到注意力分数（scores）矩阵 $ S = QK^\top \in \mathbb{R}^{n \times n} $。
        * **计算复杂度**: $ O(n^2 \cdot d) $, 因为需要计算 $ n^2 $ 个点积（总共有 $ n^2 $ 个元素），每个点积涉及 $ d $ 次乘加。
    - **计算 Softmax**：
        * 对注意力分数矩阵 $ S $ 的每一行进行 Softmax 操作，总共 $ n $ 行，每行有 $ n $ 个元素。
        * **计算复杂度**: $ O(n^2) $。
2. **应用注意力权重到值矩阵**：
    - **权重矩阵与值矩阵相乘**：
        * 注意力权重（attention_weights）矩阵 $ A \in \mathbb{R}^{n \times n} $ 与值矩阵 $ V \in \mathbb{R}^{n \times d} $ 进行矩阵相乘，得到输出矩阵 $ O = AV \in \mathbb{R}^{n \times d} $。
        * **计算复杂度**: $ O(n^2 \cdot d) $, 因为需要计算 $ n \times d $ 个点积（总共有 $ n \times d $ 个元素），每个点积涉及 $ n $ 次乘加。

**总的计算复杂度**：

$ O(n^2 \cdot d) + O(n^2) + O(n^2 \cdot d) = O(n^2 \cdot d) $

这意味着当序列长度 $ n $ 增大时，计算量会呈平方级增长，因此增加大模型的上下文并不是一个简单的事情。

#### Q2: 什么是顺序操作次数（Sequential Operations）？
**顺序操作次数**是指在处理输入序列时，**必须**按顺序执行的计算步骤数量，这些步骤无法被并行化，必须一个接一个地完成。

+ **自注意力（Self-Attention）**：所有位置的计算可以并行进行，顺序操作次数为 $ O(1) $。
+ **循环网络（Recurrent Layer）**：注意到只有 RNN 是 $ O(n) $, 因为 RNN 中每个时间步的计算依赖于前一个时间步的隐藏状态，无法并行化。
+ **卷积网络（Convolution）****和****受限自注意力（Restricted Self-Attention）**：卷积和局部注意力操作同样可以并行在所有位置上执行，顺序操作次数为 $ O(1) $。

#### Q3: 什么是最大路径长度（Maximum Path Length）？
**最大路径长度**指的是在网络中，从序列的一个位置到另一个位置传递信息所需经过的最大步骤数，这一指标反映了模型捕获长距离依赖关系的效率。

+ **自注意力（Self-Attention）**：一个 query 是和所有的 key 去做运算的，而且输出是所有 value 的加权和，所以任何两个位置之间都可以通过一次注意力操作直接交互，最大路径长度为 $ O(1) $。
+ **循环网络（Recurrent Layer）**：信息需要通过所有中间时间步传递，最大路径长度为 $ O(n) $。
+ **卷积网络（Convolution）**：可以通过堆叠多层卷积来扩大感受野，最大路径长度为 $ O(\log_k(n)) $, 其中 $ k $ 是卷积核大小。
+ **受限自注意力（Restricted Self-Attention）**：每个位置只与其相邻的 $ r $ 个位置交互，所以最大路径长度为 $ O(n/r) $。
+ **优化器**：

$ `
lrate = d_{model}^{-0.5} \cdot \min(\text{step\_num}^{-0.5}, \text{step\_num} \cdot \text{warmup\_steps}^{-1.5})
` $

$ d_{model} $ 是嵌入维度，也就是一个 token 变成 embedding 后的维度，或者说模型的宽度（model size），$ `\text{step\_num}` $ 是当前训练步数，$ `\text{warmup\_steps}` $ 是 “热身”步数，论文中 $ `\text{warmup\_steps}=4000` $, 表示在前 4000 步线性增加学习率，之后按步数平方根倒数（$ `\text{step\_num}^{-0.5}` $）逐渐减少学习率。结合下图进行理解。

### Training
#### Q1: 为什么学习率热身的时候可以刚好线性增加到 warmup_steps？
注意公式中只有 $ \text{step\_num} $ 是变量，其余为固定的常数，因此可以分三种情况来讨论：

    - 使用 Adam 优化器，参数设置为 $ \beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9} $。
    - 学习率随训练步骤变化，具体公式为：
    - 当 $ \text{step\_num} < \text{warmup\_steps} $ 时，公式第二项 $ \text{step\_num} \cdot \text{warmup\_steps}^{-1.5} $ 的值小于第一项 $ \text{step\_num}^{-0.5} $，此时学习率等于 $ d_{model}^{-0.5} \cdot \text{step\_num} \cdot \text{warmup\_steps}^{-1.5} $ ，随训练步数线性增加，直到 $ \text{step\_num} $ 达到 $ \text{warmup\_steps} $。
    - 当 $ \text{step\_num} = \text{warmup\_steps} $ 时，$ \text{step\_num} \cdot \text{warmup\_steps}^{-1.5} = \text{step\_num} \cdot \text{step\_num}^{-1.5} = \text{step\_num}^{-0.5} $，此时，学习率刚好达到峰值 $ d_{model}^{-0.5} \cdot \text{warmup\_steps}^{-0.5} $。
    - 当 $ \text{step\_num} > \text{warmup\_steps} $ 时，公式第一项 $ \text{step\_num}^{-0.5} $ 的值小于第二项 $ \text{step\_num} \cdot \text{warmup\_steps}^{-1.5} $，学习率等于 $ d_{model}^{-0.5} \cdot \text{step\_nums}^{-0.5} $，按步数平方根倒数逐渐减小。

**正则化方法**：

+ **残差 Dropout**：对每个子层（sublayer）将 dropout 应用于每个子层的输出，经过残差连接后再进行归一化（LayerNorm），见 Add & Norm 的代码实现。
    - 另外，对 embedding 和位置编码相加的地方也使用 Dropout，见位置编码的代码实现。
    - 基础模型的 Dropout 概率 $ P_{drop} $ 为 0.1。
+ **标签平滑（Label Smoothing）**: $ \epsilon_{ls} = 0.1 $, 这会增加 PPL（困惑度 perplexity），因为模型会变得更加不确定，但会提高准确性和 BLEU 分数。

#### Q2: 什么是Label Smoothing？
标签平滑（Label Smoothing）是一种与“硬标签”（hard label）相对的概念，我们通常使用的标签都是硬标签，即正确类别的概率为1，其他类别的概率为0。这种方式直观且常用，但在语言模型训练时可能会过于“极端”：在 softmax 中，只有当 logit 值无限大时，概率才能逼近 1。标签平滑的作用就是将 one-hot 转换为“软标签”，即正确类别的概率稍微小于 1，其余类别的概率稍微大于 0，形成一个更平滑的目标标签分布。具体来说，对于一个多分类问题，标签平滑后，正确类别的概率由 1 变为 $ 1 - \epsilon_{ls} $，所有类别（包括正确）再均分 $ \epsilon_{ls} $ 的概率。下面我们通过**公式**和**代码**来理解。对于一个具有 $ C $ 个类别的分类任务，假设 $ \mathbf{y} $ 是真实标签的 one-hot 编码，正确类别的概率为 1，其余类别的概率为 0：$ `
\mathbf{y} = [0, 0, \ldots, 1, \ldots, 0]
` $



应用标签平滑后，目标标签的分布 $ \mathbf{y}' $ 变为：$ `
\mathbf{y}' = (1 - \epsilon_{ls}) \cdot \mathbf{y} + \frac{\epsilon_{ls}}{C}
` $也就是说，对于正确类别 $ i $，标签平滑后的概率为：$ `
\mathbf{y}_i = 1 - \epsilon_{ls} + \frac{\epsilon_{ls}}{C}
` $对于其他类别 $ j \neq i $，标签平滑后的概率为：$ `
\mathbf{y}'_j = \frac{\epsilon_{ls}}{C}
` $标签平滑后的标签向量 $ \mathbf{y}' $ 的形式为：$ `
\mathbf{y}' = [\frac{\epsilon_{ls}}{C}, \frac{\epsilon_{ls}}{C}, \ldots, 1 - \epsilon_{ls} + \frac{\epsilon_{ls}}{C}, \ldots, \frac{\epsilon_{ls}}{C}]
` $**代码实现**：

```python
smooth = (1 - epsilon) * one_hot + epsilon / C
```

#### Q3: 什么是 PPL?
> 《[18. 模型量化技术概述及 GGUF:GGML 文件格式解析](../Guide/18.%20模型量化技术概述及%20GGUF%3AGGML%20文件格式解析.md#什么是-ppl)》
>

#### Q4: 什么是 BLEU？
**BLEU（Bilingual Evaluation Understudy）**  双语评估替换

公式：

$ \text{BLEU} = BP \cdot \exp\left( \sum_{n=1}^{N} w_n log\ p_n\right)^{\frac{1}{N}} $

首先要明确两个概念

1. **N-gram**   
用来描述句子中的一组 n 个连续的单词。比如，"Thank you so much" 中的 n-grams:
    - 1-gram: "Thank", "you", "so", "much"
    - 2-gram: "Thank you", "you so", "so much"
    - 3-gram: "Thank you so", "you so much"
    - 4-gram: "Thank you so much"  
需要注意的一点是，n-gram 中的单词是按顺序排列的，所以 "so much Thank you" 不是一个有效的 4-gram。
2. **精确度（Precision）**  
精确度是 Candidate text 中与 Reference text 相同的单词数占总单词数的比例。 具体公式如下：$ \text{Precision} = \frac{\text{Number of overlapping words}}{\text{Total number of words in candidate text}} $比如：Candidate: <u>Thank you so much</u>, ChrisReference: <u>Thank you so much</u>, my brother这里相同的单词数为4，总单词数为5，所以 $ \text{Precision} = \frac{{4}}{{5}} $但存在一个问题：
    - **Repetition** 重复Candidate: <u>Thank Thank Thank</u>Reference: <u>Thank</u> you so much, my brother此时的 $ \text{Precision} = \frac{{3}}{{3}} $

**解决方法：Modified Precision**

很简单的思想，就是匹配过的不再进行匹配。

Candidate: <u>Thank</u> Thank Thank

Reference: <u>Thank</u> you so much, my brother

$ \text{Precision}_1 = \frac{{1}}{{3}} $

+ 具体计算如下：$ `Count_{clip} = \min(Count,\ Max\_Ref\_Count)=\min(3,\ 1)=1` $$ `p_n = \frac{\sum_{\text{n-gram}} Count_{clip}}{\sum_{\text{n-gram}} Count} = \frac{1}{3}` $现在还存在一个问题：**译文过短**

Candidate: <u>Thank you</u>

Reference: <u>Thank you</u> so much, my brother

$ p_1 = \frac{{2}}{{2}} = 1 $

这里引出了 **brevity penalty**，这是一个惩罚因子，公式如下：

$ BP = \begin{cases} 1& \text{if}\ c>r\\ e^{1-\frac{r}{c}}& \text{if}\ c \leq r  \end{cases} $

其中 c 是 candidate 的长度，r 是 reference 的长度。

当候选译文的长度 c 等于参考译文的长度 r 的时候，BP = 1，当候选翻译的文本长度较短的时候，用 $ e^{1-\frac{r}{c}} $ 作为 BP 值。

回到原来的公式: $ \text{BLEU} = BP \cdot \exp\left( \sum_{n=1}^{N} w_n log\ p_n\right)^{\frac{1}{N}} $, 汇总一下符号定义：

+ $ BP $ 文本长度的惩罚因子
+ $ N $ n-gram 中 n 的最大值
+ $ w_n $ 权重
+ $ p_n $ n-gram 的精度 (precision)

### Model
#### Q1: 什么是编码器-解码器架构？
将**输入序列**编码为高维特征表示，再将这些表示解码为**输出序列**，具体数学表述如下：

+ **编码器**将输入序列 $ X = (x_1, ..., x_n) $ 映射为特征表示 $ Z = (z_1, ..., z_n) $, 这些表示实际上代表了输入的高维语义信息。
+ **解码器**基于编码器生成的表示 $ Z $, 逐步生成输出序列 $ Y = (y_1, ..., y_m) $。在每一步解码时，解码器是**自回归**（auto-regressive）的，即依赖于先前生成的符号作为输入，以生成当前符号。
    - 在第 $ t $ 步时，解码器会将上一步生成的 $ y_{t-1} $ 作为额外输入，以预测当前时间步的 $ y_t $。

#### Q2: 什么是自回归与非自回归？
![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758205493399-e7688c9e-3b22-41b3-ac6b-740af9c4a7ce.png)

**自回归（Auto-Regressive）**

**自回归生成**是指序列生成过程中，**每个新生成的 token 依赖于之前生成的 token**。这意味着生成过程是**串行的**，每一步的输入由**前面已生成的 token 组成的上下文序列**构成。例如：

+ 假设要生成一个长度为 $ T $ 的句子 $ y = (y_1, y_2, \dots, y_T) $，在生成句子 $ y $ 的过程中，首先生成 $ y_1 $，然后在生成 $ y_2 $ 时需要考虑 $ y_1 $；在生成 $ y_3 $ 时，需要考虑 $ (y_1, y_2) $，以此类推，直到生成结束符号（`<end>`）。

这种设计确保了生成过程中的连贯性和逻辑一致性，当前大多数语言模型（如 GPT 系列）都采用自回归生成的方式。

**非自回归（Non-Autoregressive）**

**非自回归生成**是一种**并行生成**的方式，**一次性生成多个甚至全部的 token**，从而显著提高生成速度，但也会**牺牲一定的生成质量**。

#### Q3: 既然输出 $ h_t $ 同样依赖于 $ h_{t-1} $, 那并行体现在哪？
虽然在**推理阶段（inference）**，生成过程看起来必须是顺序的（实际也是如此），因为每一步的输出都依赖于前一步的结果（即 $ h_t $ 依赖于 $ h_{t-1} $）。但在**训练阶段**，模型可以实现并行处理（稍微停顿一会，猜猜是如何去做的）：

**训练阶段的并行化**

在**训练阶段**，我们无需像推理时那样依赖解码器的先前输出来预测当前时间步的结果，而是使用**已知的目标序列**（Teacher Forcing）作为解码器**每个**时间步的输入，这意味着解码器的所有时间步（所有 token）可以**同时**进行预测：

+ **Teacher Forcing** 是指在训练过程中，使用真实的目标输出（ground truth）作为解码器每一步的输入，而不是依赖模型自己生成的预测结果，对于 Transformer 来说，这个目标输出就是对应的翻译文本。

> 跟先前提到的“预言家”异曲同工（或者说预言家的 IDEA 在诞生之初极有可能是受到了 Teacher Forcing 的启发，为了在推理阶段也可以并行），只是在这里模型不需要“预言”，直接对着答案“抄”就好了。
>

+ 这样一来，模型可以在所有时间步上**同时计算损失函数**。

> 结合之前的 Mask 矩阵，非自回归中的预言家以及下图进行理解。
>
> ![](https://cdn.nlark.com/yuque/0/2025/png/20367310/1758205842312-c8bf9b4b-d87a-4e4b-851f-209fb7de850f.png)
>

#### Q4:  Word Embedding（输入处理）和 Sentence Embedding（e.g., in RAG）是同一回事吗？
不是。尽管二者都将文本转换为向量表示，但它们在概念和用途上有明显不同。

+ Word Embedding
    - 以 Transformer 的输入处理中的 [Embeddings](#嵌入embeddings) 为例，用于表示单独的词或 token，将每个词映射到一个连续的向量空间。
    - **形状**：`(batch_size, seq_len, d_model)`

```python
# 输入 token ID 序列，形状为 (batch_size, seq_len)
input_tokens = [token1_id, token2_id, token3_id, ...]

# 经过嵌入层，转换为嵌入向量，形状为 (batch_size, seq_len, d_model)
input_embeddings = embedding_layer(input_tokens)
```

+ Sentence Embedding
    - 表示整个句子或文本的语义，捕获序列的整体含义，可用于下游任务（如检索增强生成 RAG、语义搜索、文本分类等）
    - **获取方式**：
        * **使用编码器生成上下文相关的嵌入**：输入序列经过编码器，生成每个 token 的上下文嵌入，形状为 `(batch_size, seq_len, d_model)`。
        * **池化策略**：将 token 的上下文嵌入聚合为一个固定大小的向量，常见方法包括：
            + **[CLS] 池化**：以 BERT 为例，可以使用 `[CLS]` token 的嵌入向量作为句子表示。
            + **平均池化**：对所有 token 的嵌入取平均值。
            + **最大池化**：取所有 token 嵌入的最大值。
        * **形状**：`(batch_size, d_model)`

```python
# 输入查询文本或文档文本
query_text = "What is RAG?"

# 对文本进行分词和编码，得到 token ID 序列
input_ids = tokenizer.encode(query_text, return_tensors='pt')

# 使用编码器将文本转换为上下文嵌入，形状为 (batch_size, seq_len, d_model)
token_embeddings = encoder(input_ids)

# 提取句子嵌入，形状为 (batch_size, d_model)
# 方法一：使用 [CLS] token 的嵌入（适用于以 [CLS] 开头的模型，如 BERT）
sentence_embedding = token_embeddings[:, 0, :]

# 方法二：平均池化
sentence_embedding = torch.mean(token_embeddings, dim=1)

# 方法三：最大池化
sentence_embedding, _ = torch.max(token_embeddings, dim=1)
```