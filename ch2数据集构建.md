### 2.1 理解 Word Embedding

每一个词都转化为n维的embedding（左图3维为例），并反映相似度关系（右图2维为例）

<center class="half">    
  <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp" width="500px"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/03.webp" width="300px">
</center>

###  2.2|2.3|2.4 分词 Tokenize | 转换TokenID | 添加特殊字符

```python
# 分词 Tokenize

"""
带括号的捕获组 (\s) 表示：
1. 用空白字符作为分隔符切割字符串
2. 同时保留这些空白字符在结果列表中(普通的input().split()方法不会保留空白字符)
"""

text = "Hello, world. Is this-- a test?"

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)
# ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/05.webp" width="400px">

```python
# 转换TokenID

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        # 将所有空白字符+标点前的空白字符去掉, 其中r'\1'表示捕获组中的标点（即删除前面的空白，保留标点本身）
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

vocab = {token: integer for integer, token in enumerate(all_words)}
tokenizer = SimpleTokenizerV1(vocab)
text = """
"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.
"""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
# [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]
# " It\' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/08.webp?123" width="500px">

- **常见特殊标记**：
  - `[BOS]`（beginning of sequence）：标记文本的起始位置
  - `[EOS]`（end of sequence）：标记文本的结束位置（通常用于拼接多个无关文本，例如两篇不同的维基百科文章或书籍）。
  - `[PAD]`（padding）：当批量训练时（批量大小 > 1），若文本长度不同，用填充标记将较短文本补齐至最长文本长度，使所有文本等长。
  - `[UNK]`（unknown）：表示词汇表中未包含的词语。
- **GPT-2 的特殊标记设计** ：
  - 无上述标记，仅使用`<|endoftext|>`：降低复杂度，作用1类似 `[EOS]`标记文本结束，作用2用于填充，由于训练时使用掩码忽略填充部分，因此填充内容无关紧要。
  - 无 `<UNK>` 标记，GPT-2 使用 **BPE（byte-pair encoding）**，将未登录词（Out-of-Vocabulary Words）分解为子词单元

```python
# 添加特殊字符

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))
# Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.
# [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
# <|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.
```

### 2.5 Byte-pair encoding

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/11.webp" width="500px">

- GPT-2 使用 BytePair 编码 (BPE) 作为其标记器
- 它允许模型**将不在其预定义词汇表中的单词分解为更小的子词单元甚至单个字符**，使其能够处理词汇表之外的单词
- 例如，如果 GPT-2 的词汇表中没有单词“unfamiliarword”，它可能会将其标记为 [“unfam”、“iliar”、“word”] 或其他子词分解，具体取决于其训练过的 BPE 合并
- 原始 BPE 标记器可以在这里找到：[https://github.com/openai/gpt-2/blob/master/src/encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- 在本章中，我们使用 OpenAI 开源 [tiktoken](https://github.com/openai/tiktoken) 库中的 BPE 标记器，该库使用 Rust 提高计算性能

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
# [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 1659, 617, 34680, 27271, 13]
strings = tokenizer.decode(integers)
print(strings)
# Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace.
```

### 2.6 滑动窗口进行数据采样

<center class="half">    
  <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/12.webp" width="400px"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/14.webp" width="400px">
</center>

```python
# 创建Dataset和DataLoader

import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
        """
        [
            [xx, xx, ...],  # index: [0, max_length]
            [xx, xx, ...],  # index: [stride, stride + max_length]
            ...
            ...
        ]  # 总共: (len(token_ids) - max_length) // stride
        """

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
# 无重叠, 因为max_length和stride都是4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
"""
Inputs:
 tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Targets:
 tensor([[  367,  2885,  1464,  1807],
        [ 3619,   402,   271, 10899],
        [ 2138,   257,  7026, 15632],
        [  438,  2016,   257,   922],
        [ 5891,  1576,   438,   568],
        [  340,   373,   645,  1049],
        [ 5975,   284,   502,   284],
        [ 3285,   326,    11,   287]])
"""
```

### 2.7 创建Embedding

```python
# Embedding layer

vocab_size = 6  # 词典大小, TokenID的数量, 每个ID都需要转化为一个Embedding vector
output_dim = 3  # Embedding vector维度

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
"""
Parameter containing:
tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]], requires_grad=True)
"""
```

- embedding layer weight 和 tokenID 的矩阵并不是相乘关系，而是通过 tokenID 来索引 embedding layer weight 矩阵的行，例如，对于 tokenID 3，我们可以通过索引 embedding layer weight 的第 4 行来获取对应的 embedding

```python
print(embedding_layer(torch.tensor([3])))
# tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/embeddings-and-linear-layers/3.png" width="450px">

```python
# Linear 实现 Embedding

idx = torch.tensor([2, 3, 1])
onehot = torch.nn.functional.one_hot(idx)
print(onehot)
"""
tensor([[0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]])
"""

torch.manual_seed(123)
linear = torch.nn.Linear(num_idx, out_dim, bias=False)
print(linear.weight)
"""
Parameter containing:
tensor([[-0.2039,  0.0166, -0.2483,  0.1886],
        [-0.4260,  0.3665, -0.3634, -0.3975],
        [-0.3159,  0.2264, -0.1847,  0.1871],
        [-0.4244, -0.3034, -0.1836, -0.0983],
        [-0.3814,  0.3274, -0.1179,  0.1605]], requires_grad=True)
"""

linear.weight = torch.nn.Parameter(embedding.weight.T)
linear(onehot.float()) == embedding(idx)
```

- 如果设置一个权重与 embedding layer weight 相同的 linear layer，并将 tokenID 转作 one-hot 编码矩阵，那么将 one-hot 编码矩阵与 linear layer 相乘，就等同于将 tokenID 作为索引来获取 embedding layer weight 矩阵的行

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/embeddings-and-linear-layers/4.png" width="450px">

### 2.8 position encoding

```python
# position_layer

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print(inputs.shape)
# torch.Size([8, 4])

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
# torch.Size([8, 4, 256])

# 一般position embedding的维度是(context_length, output_dim), 根据torch.arange(max_length)即[0, 1, 2, 3]取对应位置的embedding
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)
# torch.Size([4, 256])

# torch.Size([8, 4, 256]) + torch.Size([4, 256]), 广播至batch内每个[4, 256]
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
# torch.Size([8, 4, 256])
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/19.webp" width="500px">

### 总结代码

```python
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size, max_length, stride, 
                         shuffle=True, drop_last=True, num_workers=0):
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            drop_last=drop_last, num_workers=num_workers)
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab_size = 50257
output_dim = 256
context_length = 1024

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

batch_size = 8
max_length = 4
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=batch_size,
    max_length=max_length,
    stride=max_length
)

for batch in dataloader:
    x, y = batch

    token_embeddings = token_embedding_layer(x)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    input_embeddings = token_embeddings + pos_embeddings
    
    print(pos_embedding_layer.weight)
    print(pos_embedding_layer(torch.arange(max_length)))  # 前四行
    break
```
