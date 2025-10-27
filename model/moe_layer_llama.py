import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Optional, Tuple
import torch.nn.functional as F
from transformers import LlamaConfig

class SparseTop2MLP(nn.Module):
    def __init__(self, config, intermediate_size = 256):#None原来ffn=intermediate_size=14336->SILU->4096
        super().__init__()                               #现在intermediate_size=2048
        
        self.ffn_dim = intermediate_size
        self.hidden_dim = config.hidden_size

        self.f1 = nn.Linear(self.hidden_dim, self.ffn_dim,    bias = False)#f1: 将输入从 hidden_dim 投影到 ffn_dim。
        self.f2 = nn.Linear(self.ffn_dim,    self.hidden_dim, bias = False)#f2: 是输出层，把 ffn_dim → hidden_dim，用于还原原始维度。
        self.f3 = nn.Linear(self.hidden_dim, self.ffn_dim,    bias = False)#f3: 也是 hidden_dim → ffn_dim，但用于构造门控分支。

        self.act = nn.SiLU()# transformer 中常用的非线性函数SiLU(x)=x⋅sigmoid(x)

    def forward(self, hidden_state):#hidden_state 是来自上一层的输出，形状通常是 [batch_size, seq_len, hidden_dim]

        x = self.act(self.f1(hidden_state) * self.f3(hidden_state))
        x = self.f2(x)
        return x

class MoeBlock_RS(nn.Module):

    def __init__(self, config, cluster_index_list, dataset_num):
        super().__init__()
        self.ffn_dim = 1280            # 每个专家内部隐层 (原 14336，这里降维节省算力)
        self.hidden_dim = config.hidden_size  # Llama-3 hidden size, 在 Llama-3 8B 里，Transformer 的主隐藏维度 hidden_size 就是 4096->6个专家(在显存范围内)
        self.num_experts = 12                       # K = 12 个专家,因为4096改成2048了
        self.top_k = 2                              # 每个 token 只走 2 个专家 (稀疏路由)
        self.num_cluster = 4                        # 用户-物品聚类数量 = 5
        
        # ------------------------ 1️⃣ 门控 (Gate) ----------------
        #   gate0, gate1, … gate4 分别对应 5 个聚类的独立软路由器
        #   每个门控：hidden → num_experts logits
        self.gate = nn.ModuleDict({f"gate{i}": nn.Linear(self.hidden_dim, self.num_experts, bias = False) for i in range(self.num_cluster)})
        # ------------------------ 2️⃣ 专家 (Experts) --------------
        #   这里用作者定义的 SparseTop2MLP：内部是 feed-forward MLP
        self.experts = nn.ModuleList([SparseTop2MLP(config) for _ in range(self.num_experts)])
        # ------------------------ 3️⃣ 运行时辅助变量 --------------
        self.cluster_index_list = cluster_index_list    # 每条样本所属聚类（离线预计算）
        self.foward_count = 0   
        self.cluster_index_count = 0                    # 游标：指向当前样本的聚类索引游标 / 指针，记录“当前 MoE 层已经处理到第几条样本”
        self.dataset_num = dataset_num                  # 数据集大小（可做归一化等）
    '''
        input: 
            hidden_state: Transformer FFN
            cluster_index: index for choose which gate to use
                ex: cluster 0 : gate 0
                    cluster 1 : gate 1
                    ...
                    cluster n : gate n
                shape: [batch_size]
    '''
    # -------------------------------------------------------------------------
    #  forward(hidden_states)  
    #     hidden_states: (B, L, H) —— Transformer 的 token 表示
    #hidden_states 并不是在 MoeBlock_RS 里生成的，而是 作为参数由上一层 Transformer 传进来的。
    # -------------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        router_logits_list = []
        # ---------- 1. 计算当前 batch 每个样本的路由 logits ----------
        for i,idx in enumerate(range(batch_size)):
            # ❶ 取本样本在聚类列表中的索引
            gate_index = self.cluster_index_list[self.cluster_index_count-1]#在进入 for i in range(batch_size): 循环之前，游标已经被 +1。
            # ❷ 用对应 gate_i 生成 (L, H) @ (H, num_experts)->(L, num_experts) logits;用对应的门控 gateX 把该样本的 每个 token 表示 映射成 num_experts 个分数。 shape=(L,12)
            #接收当前 token 的隐藏向量，输出对 全部专家 的打分，再挑若干个最合适的专家去处理。
            routing_logits = self.gate['gate{}'.format(gate_index)](hidden_states[i])
            router_logits_list.append(routing_logits)
        # concat → (B*L, num_experts),方便一次性对所有 token 做 softmax/topk
        router_logits = torch.stack(router_logits_list).view(-1, self.num_experts)
        # reshape hidden 为 (B*L, H) 方便逐 token 处理
        hidden_states = hidden_states.view(-1, hidden_dim)
        # ---------- 2. softmax 得到路由概率 ----------
        routing_weights = F.softmax(router_logits, dim = -1)
        # ---------- 3. 取 Top-2 专家 ----------
        # select top2 experts
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim = -1)   # (B*L, 2)
        # fusing weight && add  归一化权重，使两专家权重相加 =1
        routing_weights = routing_weights / torch.sum(routing_weights, dim = -1, keepdim = True).to(hidden_states.dtype)
        # ---------- 4. 准备输出占位 tensor -------
        #  init maxtrix to save result，全零 张量，shape=(B·L, H)
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),dtype=hidden_states.dtype,device=hidden_states.device
        )
        # ---------- 5. one-hot 掩码，把 token 分配到对应专家 ----------
        # expert_mask: (num_experts, top_k, B*L)
        # for efficiency, calculate the result one time using the mask
        expert_mask = nn.functional.one_hot(selected_experts, num_classes = self.num_experts)
        #expert_mask[e]  (top_k=2, B·L=20)
         #r=0  ▒ 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 ▒
         #r=1  ▒ 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0 ▒
        #torch.where → idx=[0,1, …]   top_x=[1,3, …]
        # [20,2,8] ---> [8,2,20],这里的8应该改成12，因为有12个专家
        expert_mask = expert_mask.permute(2, 1, 0)
        # ---------- 6. 逐专家计算，并把结果累加到 final_hidden ----------
        for expert_index in range(self.num_experts):
            expert_layer = self.experts[expert_index]
            idx, top_x = torch.where(expert_mask[expert_index])# 哪些 token 选了该专家
            top_x_list = top_x.tolist()                   # 对应在 hidden_states 的行号
            idx_list = idx.tolist()                       # 0 或 1（两条路由中的哪个）
            current_state = hidden_states[None,top_x_list].reshape(-1, hidden_dim)  # 收集属于该专家的一堆 token 表示，shape (Nᵗ, H)。
            #① 批量经过专家 MLP；# routing_weights[top_x_list,idx_list, None] 取对应的概率 w₀，unsqueeze 成 (Nᵗ, 1)，才能与 (Nᵗ, H) 做逐行乘法
            #② 乘上路由概率做加权。
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list,idx_list, None] # 专家前向
            current_hidden_states = current_hidden_states.to(hidden_states.dtype)
            #- 如果一个 token 走了 2 个专家，index_add_ 会在同一行把两路结果相加（符合论文里的 加权求和）。
            #例子output_token3 = w₀ · y₀ + w₁ · y₁= 0.7·[10,10]  +  0.3·[20,20]= [13, 13]
            final_hidden_states.index_add_(0, top_x, current_hidden_states)#dim：在哪个维度做累加，这里 0 = 行，index：行/列索引张量，source：要加进去的张量，形状与 index 对应维度长度一致
        final_hidden_states = final_hidden_states.reshape(batch_size,seq_len,hidden_dim)

        return final_hidden_states


if __name__ == '__main__':
    import random 
    config = LlamaConfig()
    cluster_index_list = [random.randint(0, 4) for _ in range(10)]
    print(f'index{cluster_index_list}')
    moe_block = MoeBlock_RS(config, cluster_index_list)
    test_tensor = torch.randn(5,10,4096)
    out = moe_block(test_tensor)
    # print(out)
    print(out.shape)
