# %% [markdown]
# # 自动写诗      
# 
# ### 作者：郑之杰
# 
# 首先导入必要的库：

# %%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# %% [markdown]
# # 加载数据集
# 
# 本次实验的数据来自chinese-poetry：https://github.com/chinese-poetry/chinese-poetry
# 
# 实验提供预处理过的数据集，含有57580首唐诗，每首诗限定在125词，不足125词的以```<s>```填充。数据集以npz文件形式保存，包含三个部分：
# - （1）data: 诗词数据，将诗词中的字转化为其在字典中的序号表示。
# - （2）ix2word: 序号到字的映射
# - （3）word2ix: 字到序号的映射
# 
# 预处理数据集的下载：[点击下载](https://yun.sfo2.digitaloceanspaces.com/pytorch_book/pytorch_book/tang.npz)

# %%
def prepareData():
    
    # 读入预处理的数据
    datas = np.load("tang.npz",allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    
    # 转为torch.Tensor
    data = torch.from_numpy(data)
    dataloader = DataLoader(data,
                         batch_size = 16,
                         shuffle = True,
                         num_workers = 2
                         )
    
    return dataloader, ix2word, word2ix

# %%
dataloader, ix2word, word2ix = prepareData()

# %% [markdown]
# # 构建模型
# 
# 模型包括Embedding层、LSTM层和输出层。

# %%
class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=3, dropout=0.5)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()

        if hidden is None:
            h_0 = input.data.new(4, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(4, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        embeds = self.embedding(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.dropout(output)
        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden

# %% [markdown]
# # 训练模型

# %%
# 设置超参数
learning_rate = 5e-3       # 学习率
embedding_dim = 128        # 嵌入层维度
hidden_dim = 256           # 隐藏层维度
model_path = None          # 预训练模型路径
epochs = 4                 # 训练轮数
verbose = True             # 打印训练过程
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
def train(dataloader, ix2word, word2ix):

    # 配置模型，是否继续上一次的训练
    model = PoetryModel(len(word2ix), embedding_dim, hidden_dim)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # 设置损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义训练过程
    for epoch in range(epochs):
        for batch_idx, data in enumerate(dataloader):
            data = data.long().transpose(1, 0).contiguous()
            data = data.to(device)
            input, target = data[:-1, :], data[1:, :]
            output, _ = model(input)
            loss = criterion(output, target.view(-1))
            
            if batch_idx % 900 == 0 & verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data[1]), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item()))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')

# %%
train(dataloader, ix2word, word2ix)

# %% [markdown]
# # 生成唐诗
# 
# 给定几个词，根据这几个词接着生成一首完整的唐诗。

# %%
# 设置超参数
model_path = 'model.pth'        # 模型路径
start_words = '湖光秋月两相和'  # 唐诗的第一句
max_gen_len = 125                # 生成唐诗的最长长度

# %%
def generate(start_words, ix2word, word2ix):

    # 读取模型
    model = PoetryModel(len(word2ix), embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # 读取唐诗的第一句
    results = list(start_words)
    start_word_len = len(start_words)
    
    # 设置第一个词为<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.to(device)
    hidden = None

    # 生成唐诗
    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        # 读取第一句
        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        # 生成后面的句子
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        # 结束标志
        if w == '<EOP>':
            del results[-1]
            break
            
    return results

# %%
results = generate(start_words, ix2word, word2ix)
print(results)

# %% [markdown]
# # 生成藏头诗
# 
# 

# %%
# 设置超参数
model_path = 'model.pth'                 # 模型路径
start_words_acrostic = '湖光秋月两相和'  # 唐诗的“头”
max_gen_len_acrostic = 125               # 生成唐诗的最长长度

# %%
def gen_acrostic(start_words, ix2word, word2ix):

    # 读取模型
    model = PoetryModel(len(word2ix), embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # 读取唐诗的“头”
    results = []
    start_word_len = len(start_words)
    
    # 设置第一个词为<START>
    input = (torch.Tensor([word2ix['<START>']]).view(1, 1).long())
    input = input.to(device)
    hidden = None

    index = 0            # 指示已生成了多少句
    pre_word = '<START>' # 上一个词

    # 生成藏头诗
    for i in range(max_gen_len_acrostic):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]

        # 如果遇到标志一句的结尾，喂入下一个“头”
        if (pre_word in {u'。', u'！', '<START>'}):
            # 如果生成的诗已经包含全部“头”，则结束
            if index == start_word_len:
                break
            # 把“头”作为输入喂入模型
            else:
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
                
        # 否则，把上一次预测作为下一个词输入
        else:
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w
        
    return results

# %%
results_acrostic = gen_acrostic(start_words_acrostic, ix2word, word2ix)
print(results_acrostic)


