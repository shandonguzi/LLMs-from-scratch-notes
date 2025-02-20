### 3.1 长序列建模问题

- 由于源语言和目标语言之间的语法结构差异，逐字翻译文本是不可行的（左图）
- 在引入 Transformer 模型之前，编码器-解码器 RNN 通常用于机器翻译任务（右图），在此设置中，编码器处理来自源语言的标记序列，使用隐藏状态（神经网络中的一种中间层）来生成整个输入序列的压缩表示

<center class="half">    
  <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/03.webp" width="400px">
  <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/04.webp" width="400px">
</center> 

### 3.2 使用注意力机制捕捉数据间依赖

- 通过注意力机制，网络的文本生成解码器部分能够有选择地访问所有输入标记，这意味着在生成特定输出标记时，某些输入标记比其他输入标记更重要
- transformer 中的自注意力是一种旨在增强输入表示的技术，它使序列中的每个位置能够与同一序列中其他每个位置互动并确定它们的相关性

<center class="half">    
  <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/05.webp" width="400px">
  <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/06.webp" width="400px">
</center> 

**下述章节脉略**

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/02.webp" width="700px">

### 3.3 无可训练权重的简单注意力机制（Simplified self-attention）

##### 3.3.1 使用自注意力机制关注输入的不同部分

- 假设我们得到一个输入序列 $x^{(1)}$ 到 $x^{(T)}$，输入是一个文本（例如，像 "Your journey starts with one step" 这样的句子），它已经按照第 2 章中的描述转换为 token 嵌入，例如，$x^{(1)}$ 是一个 d 维向量，表示单词 "Your"，依此类推
- **目标**：为 $x^{(1)}$ 到 $x^{(T)}$ 中的每个输入序列元素 $x^{(i)}$ 计算上下文向量 $z^{(i)}$（其中 $z$ 和 $x$ 具有相同的维度），上下文向量$z^{(i)}$ 是输入 $x^{(1)}$ 到 $x^{(T)}$ 的加权和（以$z^{(2)}$为例）

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/07.webp" width="500px">

- 按照惯例，未标准化的注意力权重被称为**“attention scores”**，而标准化的注意力分数（总和为 1）被称为**“attention weights”**

**步骤 1：**计算非规范化注意力得分 $\omega$

假设我们使用第二个输入标记作为查询（**query**），即 $q^{(2)} = x^{(2)}$，我们通过点积计算非规范化注意力得分：

$\omega_{21} = x^{(1)} q^{(2)\top}$

$\omega_{22} = x^{(2)} q^{(2)\top}$

$\omega_{23} = x^{(3)} q^{(2)\top}$

...

$\omega_{2T} = x^{(T)} q^{(2)\top}$

以上，$\omega$ 是希腊字母"omega"，用于表示非规范化注意力得分

$\omega_{21}$ 中的下标"21"表示输入序列元素 2 是用作针对输入序列元素 1 的查询

```python
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]  # 2nd input token is the query
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    # dot product (transpose not necessary here since they are 1-dim vectors)
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)
# tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/08.webp" width="500px">

**步骤 2：** 将未标准化的注意力得分 ("omegas",  $\omega$) 标准化，使它们的总和为 1

这是一个将未标准化的注意力得分标准化为总和为 1 的简单方法（一种惯例，对解释有用，对训练稳定性很重要）

然而，在实践中，**使用 softmax 函数进行归一化**很常见，也是推荐的，因为它更善于处理极值，并且在训练期间具有更理想的梯度特性。这是一个用于缩放的 softmax 函数的简单实现，它还将向量元素归一化，使它们的总和为 1。由于溢出和下溢问题，上述简单实现可能会因大或小输入值而遭受数值不稳定问题，因此，在实践中，建议改用 **softmax 的 PyTorch 实现**，该实现已针对性能进行了高度优化。

```python
# 归一化简单实现
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())
"""
Attention weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
Sum: tensor(1.0000)
"""

# Softmax 归一化
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())
"""
Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
Sum: tensor(1.)
"""

# Pytorch Softmax 归一化（优化）
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())
"""
Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
Sum: tensor(1.)
"""
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/09.webp" width="500px">

**步骤 3**：通过将嵌入的输入标记 $x^{(i)}$ 与注意力权重相乘来计算上下文向量 $z^{(2)}$，并对得到的向量求和：

```python
query = inputs[1]  # 2nd input token is the query
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)
# tensor([0.4419, 0.6515, 0.5683])
```

##### 3.3.2 计算所有输入Token的注意力权重（3.3.1仅计算$z^{(2)}$）

```python
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)
"""
tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
        [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
        [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
        [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
        [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
"""

# 效果同上
attn_scores = inputs @ inputs.T
print(attn_scores)

# 针对最后一维进行 Softmax
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
"""
tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
        [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
        [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
        [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
        [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
        [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])
"""

# 计算 context vector
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
"""
tensor([[0.4421, 0.5931, 0.5790],
        [0.4419, 0.6515, 0.5683],  # 与3.3.1步骤3中的context_vec_2一样
        [0.4431, 0.6496, 0.5671],
        [0.4304, 0.6298, 0.5510],
        [0.4671, 0.5910, 0.5266],
        [0.4177, 0.6503, 0.5645]])
"""
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/11.webp" width="500px">

### 3.4 可训练权重的自注意力机制（Self-attention，也叫scaled dot-product attention）

##### 3.4.1 一步步计算attention weights

逐步实现自注意力机制，我们首先引入三个训练权重矩阵 $W_q$、$W_k$ 和 $W_v$，这三个矩阵用于通过矩阵乘法将嵌入的输入标记 $x^{(i)}$ 投影到$query、key、value$向量中：

- $query：q^{(i)} = W_q \,x^{(i)}$
- $key：k^{(i)} = W_k \,x^{(i)}$
- $value：v^{(i)} = W_v \,x^{(i)}$

```python
# 以 query_2 为例

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1]  # second input element
d_in = inputs.shape[1]  # the input embedding size, d=3
d_out = 2  # the output embedding size, d=2

torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out))
W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

query_2 = x_2 @ W_query  # _2 because it's with respect to the 2nd input element
keys = inputs @ W_key
attn_scores_2 = query_2 @ keys.T  # All attention scores for given query
print(attn_scores_2)
# tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])

# 归一化 /sqrt{d_k}
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
# tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])

# attn_weights_2 加权 values
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
# tensor([0.3061, 0.8210])
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/16.webp" width="700px">

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/17.webp" width="700px">

##### 3.4.2 SelfAttention类

```python
import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
"""
tensor([[0.2996, 0.8053],
        [0.3061, 0.8210],
        [0.3058, 0.8203],
        [0.2948, 0.7939],
        [0.2927, 0.7891],
        [0.2990, 0.8040]], grad_fn=<MmBackward0>)
"""
```

我们可以使用 PyTorch 的 $nn.Linear$ 简化上述实现，如果我们禁用 $bias$，与$nn.Parameter(torch.rand(...))$ 相比，使用 $nn.Linear$ 的另一大优势是 $nn.Linear$ 具有**更好的权重初始化方案（Xavier 或 Kaiming 初始化）**，从而可以实现更稳定的模型训练

```python
import torch.nn as nn

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
"""
tensor([[-0.0739,  0.0713],
        [-0.0748,  0.0703],
        [-0.0749,  0.0702],
        [-0.0760,  0.0685],
        [-0.0763,  0.0679],
        [-0.0754,  0.0693]], grad_fn=<MmBackward0>)
"""
```

### 3.5 隐藏未来词的因果注意力（casual attention）

在因果注意力中，对角线上方的注意力权重被掩盖，确保对于任何给定的输入，LLM 在使用注意力权重计算上下文向量时无法利用未来的标记

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/19.webp" width="500px">

##### 3.5.1 简单的casual attention mask

```python
# casual attention mask

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs) 
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)
"""
tensor([[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510],
        [0.2041, 0.1659, 0.1662, 0.1496, 0.1665, 0.1477],
        [0.2036, 0.1659, 0.1662, 0.1498, 0.1664, 0.1480],
        [0.1869, 0.1667, 0.1668, 0.1571, 0.1661, 0.1564],
        [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.1585],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<SoftmaxBackward0>)
"""

# 通过 PyTorch 的 tril 函数创建一个掩码，将主对角线下方的元素（包括对角线本身）设置为 1，将主对角线上方的元素设置为 0
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
"""
tensor([[1., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1.]])
"""

# 对应相乘即可
masked_simple = attn_weights*mask_simple
print(masked_simple)
"""
tensor([[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
        [0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
        [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<MulBackward0>)
"""

# re-normalize
"""
如果keepdim=True，那么输出的张量在指定的dim上保留尺寸1，这样做的目的是为了方便广播，每一行会除以相同的值归一化, [6, 6]除以[6, 1], [6, 1]保证广播后的[6, 6]每一行是相同的值
如果keepdim=False，那么输出的张量在指定的dim上不保留尺寸1，每一列会除以相同的值, [6, 6]除以[6], [6]会先变为[1, 6]在广播为[6, 6], 此时每一列是相同的值
"""
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
"""
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
        [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
        [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<DivBackward0>)
"""

# 更简洁的实现方式(此处是triu上边是tril, diagonal=1表示向上偏移一个单位再填充1)
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print(mask)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)
"""
tensor([[0., 1., 1., 1., 1., 1.],
        [0., 0., 1., 1., 1., 1.],
        [0., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0.]])
tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
        [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
        [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
        [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
        [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
        [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
       grad_fn=<MaskedFillBackward0>)
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
        [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
        [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<SoftmaxBackward0>)
"""
```

##### 3.5.2 masking with dropout

- $dropout$ 可以减少过拟合，更常见的做法是在**计算 $attention\_weights$ 后接 $dropout$**，这可能导致一行之和不再为1
- $dropout\ rate$ 一般为 0.1 / 0.2
- **计算后的其他值等比例放大 $1 / (1 - dropout\_rate)$**

```python
# dropout rate 以 0.5 为例

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)  # dropout rate of 50%
example = torch.ones(6, 6)  # create a matrix of ones
print(dropout(example))
"""
tensor([[2., 2., 2., 2., 2., 2.],
        [0., 2., 0., 0., 0., 0.],
        [0., 0., 2., 0., 2., 0.],
        [2., 2., 0., 0., 0., 2.],
        [2., 0., 0., 0., 0., 2.],
        [0., 2., 0., 0., 0., 0.]])
"""
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/22.webp" width="500px">

##### 3.5.3 Casual attention类

```python
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        # num_tokens是为了防止不足context_length的情况, 比如普通batch[2, 6, 3], 最后一个batch[1, 5, 3]
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)  # New

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
d_in, d_out = 3, 2
batch = torch.stack((inputs, inputs), dim=0)  # inputs[6, 3], batch[2, 6, 3]
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
"""
tensor([[[-0.4519,  0.2216],
         [-0.5874,  0.0058],
         [-0.6300, -0.0632],
         [-0.5675, -0.0843],
         [-0.5526, -0.0981],
         [-0.5299, -0.1081]],

        [[-0.4519,  0.2216],
         [-0.5874,  0.0058],
         [-0.6300, -0.0632],
         [-0.5675, -0.0843],
         [-0.5526, -0.0981],
         [-0.5299, -0.1081]]], grad_fn=<UnsafeViewBackward0>)
context_vecs.shape: torch.Size([2, 6, 2])
"""
```

### 3.6 多头注意力（Multi-head attention）

##### 3.6.1 堆叠多个单头

```python
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        # 最后一维拼接, out_dim_final = num_heads * d_out
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)

context_length = batch.shape[1]  # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
"""
tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],
         [-0.5874,  0.0058,  0.5891,  0.3257],
         [-0.6300, -0.0632,  0.6202,  0.3860],
         [-0.5675, -0.0843,  0.5478,  0.3589],
         [-0.5526, -0.0981,  0.5321,  0.3428],
         [-0.5299, -0.1081,  0.5077,  0.3493]],

        [[-0.4519,  0.2216,  0.4772,  0.1063],
         [-0.5874,  0.0058,  0.5891,  0.3257],
         [-0.6300, -0.0632,  0.6202,  0.3860],
         [-0.5675, -0.0843,  0.5478,  0.3589],
         [-0.5526, -0.0981,  0.5321,  0.3428],
         [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)
context_vecs.shape: torch.Size([2, 6, 4])
"""
```

##### 3.6.2 权重分割多头注意力

```python
# 手写MultiHead版本

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a $num_heads$ dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # contiguous确保tensor在内存中物理连续(之前的transpose操作可能导致不连续), 再使用view不会报错
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
"""
tensor([[[0.3190, 0.4858],
         [0.2943, 0.3897],
         [0.2856, 0.3593],
         [0.2693, 0.3873],
         [0.2639, 0.3928],
         [0.2575, 0.4028]],

        [[0.3190, 0.4858],
         [0.2943, 0.3897],
         [0.2856, 0.3593],
         [0.2693, 0.3873],
         [0.2639, 0.3928],
         [0.2575, 0.4028]]], grad_fn=<ViewBackward0>)
context_vecs.shape: torch.Size([2, 6, 2])
"""
```

```python
# nn.MultiheadAttention版本, 手动生成QKV(实际不需要)

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        self.d_out = d_out
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=d_out, num_heads=num_heads, dropout=dropout)

        # 方便演示所以设置, 实际nn.MultiheadAttention内部实现已经完成了对 query、key 和 value 的线性变换
        # 因此，可以直接传入 x，而不需要自己预先生成这三个不同的变换中QKV均是x
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Apply the projection layers to the input x
        queries = self.W_query(x)  # Shape: (b, num_tokens, d_out)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Reshape the input for nn.MultiheadAttention, which expects (sequence_length, batch_size, embedding_dim)
        queries = queries.transpose(0, 1)  # Shape: (num_tokens, b, d_out)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)

        # Apply causal mask if needed
        attn_mask = self.mask[:num_tokens, :num_tokens].bool()

        # Get the attention output
        attn_output, _ = self.mha(queries, keys, values, attn_mask=attn_mask)

        # Return the output of the attention
        output = attn_output.transpose(0, 1)  # Back to (b, num_tokens, d_out)
        output = self.out_proj(output)

        return output

# Example usage
torch.manual_seed(123)

# Assuming $batch$ is a tensor of shape (b, num_tokens, d_in)
batch_size, context_length, d_in = batch.shape
d_out = 2

model = MultiHeadAttention(d_in, d_out, context_length, dropout=0.1, num_heads=2)
output = model(batch)
print(output)
print("output.shape:", output.shape)
```

```python
# nn.MultiheadAttention版本, x直接传入mha, 默认输入输出的dim一样, 所以没有用到d_in

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, need_weights=True):
        super().__init__()

        self.context_length = context_length
        self.d_out = d_out
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=d_out, num_heads=num_heads, 
                                         dropout=dropout, batch_first=True)

        self.need_weights = need_weights
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        # attn_mask broadcasting will handle batch_size dimension implicitly
        attn_output, _ = self.mha(
            x, x, x, attn_mask=attn_mask, need_weights=self.need_weights
        )
        
        output = self.out_proj(attn_output)

        return output

# Example usage
torch.manual_seed(123)

# Assuming $batch$ is a tensor of shape (b, num_tokens, d_in)
batch_size, context_length, d_in = batch.shape
d_out = 3

model = MultiHeadAttention(d_in, d_out, context_length, dropout=0.1, num_heads=3)
output = model(batch)
print(output)
print("output.shape:", output.shape)
```

**`nn.MultiHeadAttention` 的三x版本比手撕 `MultiHeadAttention` 版本的参数量多 60w 左右，在 GPT-2标准版的实现中体现为最终多了 7million 个参数（12 * 60w）**

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/26.webp" width="500px">

### 番外——缓冲区

本质上，PyTorch 缓冲区是与 PyTorch 模块或模型相关的张量属性，类似于参数，但与参数不同，缓冲区在训练期间不会更新。

PyTorch 中的缓冲区在处理 GPU 计算时特别有用，因为它们需要与模型的参数一起在设备之间传输（例如从 CPU 到 GPU）。与参数不同，缓冲区不需要梯度计算，但它们仍然需要位于正确的设备上以确保所有计算都正确执行。

##### 无 buffer

```python
import torch
import torch.nn as nn

class CausalAttentionWithoutBuffers(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        # self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec

# 无cuda(win)/mps(mac)版本, 可以正常运行
torch.manual_seed(123)
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]
d_in = inputs.shape[1]
d_out = 2
ca_without_buffer = CausalAttentionWithoutBuffers(d_in, d_out, context_length, 0.0)
with torch.no_grad():
    context_vecs = ca_without_buffer(batch)
print(context_vecs)
"""
tensor([[[-0.4519,  0.2216],
         [-0.5874,  0.0058],
         [-0.6300, -0.0632],
         [-0.5675, -0.0843],
         [-0.5526, -0.0981],
         [-0.5299, -0.1081]],

        [[-0.4519,  0.2216],
         [-0.5874,  0.0058],
         [-0.6300, -0.0632],
         [-0.5675, -0.0843],
         [-0.5526, -0.0981],
         [-0.5299, -0.1081]]])
"""

# 转移至cuda/mps运行
print("Machine has GPU:", torch.backends.mps.is_available())
batch = batch.to("mps")
ca_without_buffer.to("mps")
"""
Machine has GPU: True
CausalAttentionWithoutBuffers(
  (W_query): Linear(in_features=3, out_features=2, bias=False)
  (W_key): Linear(in_features=3, out_features=2, bias=False)
  (W_value): Linear(in_features=3, out_features=2, bias=False)
  (dropout): Dropout(p=0.0, inplace=False)
)
"""

# 直接运行报错
with torch.no_grad():
    context_vecs = ca_without_buffer(batch)
print(context_vecs)
# RuntimeError: expected self and mask to be on the same device, but got mask on cpu and self on mps:0

# mask没有被移到cuda/mps, 因为他不是Pytorch Parameter
print("W_query.device:", ca_without_buffer.W_query.weight.device)
print("mask.device:", ca_without_buffer.mask.device)
"""
W_query.device: mps:0
mask.device: cpu
"""

# 需要手动移到cude/mps
ca_without_buffer.mask = ca_without_buffer.mask.to("mps")
print("mask.device:", ca_without_buffer.mask.device)
# mask.device: mps:0
```

**与常规张量相比，PyTorch 缓冲区的另一个优势是它们被包含在模型的 $state\_dict$ 中，且在修改 $mask$ 后进行的模型保存与加载会保存修改后的结果**

```python
ca_without_buffer.state_dict()
ca_with_buffer.state_dict()
"""
OrderedDict([('W_query.weight',
              tensor([[-0.2354,  0.0191, -0.2867],
                      [ 0.2177, -0.4919,  0.4232]], device='mps:0')),
             ('W_key.weight',
              tensor([[-0.4196, -0.4590, -0.3648],
                      [ 0.2615, -0.2133,  0.2161]], device='mps:0')),
             ('W_value.weight',
              tensor([[-0.4900, -0.3503, -0.2120],
                      [-0.1135, -0.4404,  0.3780]], device='mps:0'))])
OrderedDict([('mask',
              tensor([[0., 1., 1., 1., 1., 1.],
                      [0., 0., 1., 1., 1., 1.],
                      [0., 0., 0., 1., 1., 1.],
                      [0., 0., 0., 0., 1., 1.],
                      [0., 0., 0., 0., 0., 1.],
                      [0., 0., 0., 0., 0., 0.]], device='mps:0')),
             ('W_query.weight',
              tensor([[-0.1362,  0.1853,  0.4083],
                      [ 0.1076,  0.1579,  0.5573]], device='mps:0')),
             ('W_key.weight',
              tensor([[-0.2604,  0.1829, -0.2569],
                      [ 0.4126,  0.4611, -0.5323]], device='mps:0')),
             ('W_value.weight',
              tensor([[ 0.4929,  0.2757,  0.2516],
                      [ 0.2377,  0.4800, -0.0762]], device='mps:0'))])
"""

# 修改buffer, 保存模型可保留修改结果
ca_with_buffer.mask[ca_with_buffer.mask == 1.] = 2.
torch.save(ca_with_buffer.state_dict(), "model.pth")
new_ca_with_buffer = CausalAttentionWithBuffer(d_in, d_out, context_length, 0.0)
new_ca_with_buffer.load_state_dict(torch.load("model.pth"))
new_ca_with_buffer.mask
"""
tensor([[0., 2., 2., 2., 2., 2.],
        [0., 0., 2., 2., 2., 2.],
        [0., 0., 0., 2., 2., 2.],
        [0., 0., 0., 0., 2., 2.],
        [0., 0., 0., 0., 0., 2.],
        [0., 0., 0., 0., 0., 0.]])
"""


# 修改非buffer的mask, 不可保留修改结果
ca_without_buffer.mask[ca_without_buffer.mask == 1.] = 2.
torch.save(ca_without_buffer.state_dict(), "model.pth")
new_ca_without_buffer = CausalAttentionWithoutBuffers(d_in, d_out, context_length, 0.0)
new_ca_without_buffer.load_state_dict(torch.load("model.pth"))
new_ca_without_buffer.mask
"""
tensor([[0., 1., 1., 1., 1., 1.],
        [0., 0., 1., 1., 1., 1.],
        [0., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0.]])
"""
```
