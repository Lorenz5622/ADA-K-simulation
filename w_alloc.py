class ExpertSelector(nn.Module):
    """
    专家选择器，用于根据hidden_states选择最佳专家
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        
        # 创建路由线性层，将hidden_states映射到专家概率分布
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化路由器权重
        """
        # 使用较小的标准差初始化，避免极端概率
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
    
    def forward(self, hidden_states):
        """
        根据hidden_states计算专家选择概率并采样
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            probs: 专家选择概率分布 [batch_size, seq_len, num_experts]
            selected_experts: 采样得到的专家索引 [batch_size, seq_len]
        """
        # 计算每个专家的得分
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        
        # 通过softmax获得概率分布
        probs = torch.nn.functional.softmax(router_logits, dim=-1)
        
        # 从概率分布中采样获得专家索引
        # 方法1: 多项式采样
        selected_experts = torch.multinomial(probs.view(-1, self.num_experts), 1).view(
            hidden_states.shape[:-1]
        )
        
        return probs, selected_experts
    
    def get_top_k_experts(self, hidden_states, k=2):
        """
        获取top-k专家而不是采样
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            k: 选择的专家数量
            
        Returns:
            topk_probs: top-k专家的概率 [batch_size, seq_len, k]
            topk_indices: top-k专家的索引 [batch_size, seq_len, k]
        """
        router_logits = self.router(hidden_states)
        probs = torch.nn.functional.softmax(router_logits, dim=-1)
        
        # 获取top-k专家
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
        
        return topk_probs, topk_indices
    
    def compute_routing_entropy(self, hidden_states):
        """
        计算路由熵，衡量专家选择的多样性
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            entropy: 标量，路由熵值
        """
        _, probs = self.forward(hidden_states)
        
        # 计算熵: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return torch.mean(entropy)

# 使用示例
class EnhancedSwitchMLP(nn.Module):
    """
    增强版SwitchMLP，使用ExpertSelector进行专家选择
    """
    def __init__(self, config, layer_idx):
        super(EnhancedSwitchMLP, self).__init__()
        self.layer_num = layer_idx
        self.use_switch = (layer_idx % config.expert_frequency) == 0
        
        if self.use_switch:
            # 使用新的专家选择器
            self.expert_selector = ExpertSelector(config)
            self.experts = torch.nn.ModuleList()
            
            for i in range(config.num_experts):
                self.experts.append(LlamaMLP(config.hidden_size, config.intermediate_size, config.hidden_act))
        else:
            self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
    
    def forward(self, hidden_states, use_sampling=True):
        if not self.use_switch:
            return self.mlp(hidden_states)
        
        if use_sampling:
            # 使用采样方式选择专家
            probs, selected_experts = self.expert_selector(hidden_states)
            
            # 根据采样的专家进行处理
            output = self.process_with_selected_experts(hidden_states, selected_experts)
        else:
            # 使用top-k方式选择专家
            topk_probs, topk_indices = self.expert_selector.get_top_k_experts(hidden_states, k=2)
            
            # 根据top-k专家进行处理
            output = self.process_with_topk_experts(hidden_states, topk_probs, topk_indices)
            
        return output
    
    def process_with_selected_experts(self, hidden_states, selected_experts):
        """
        根据采样选择的专家处理hidden_states
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 重塑为二维进行处理
        hidden_flat = hidden_states.view(-1, hidden_size)  # [batch*seq_len, hidden_size]
        selected_flat = selected_experts.view(-1)  # [batch*seq_len]
        
        output = torch.zeros_like(hidden_flat)
        
        # 为每个专家处理对应的tokens
        for expert_idx, expert in enumerate(self.experts):
            mask = (selected_flat == expert_idx)
            if mask.any():
                expert_output = expert(hidden_flat[mask])
                output[mask] = expert_output
        
        return output.view(batch_size, seq_len, hidden_size)
    
    def process_with_topk_experts(self, hidden_states, topk_probs, topk_indices):
        """
        根据top-k专家处理hidden_states
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        k = topk_probs.shape[-1]
        
        hidden_flat = hidden_states.view(-1, hidden_size)
        probs_flat = topk_probs.view(-1, k)
        indices_flat = topk_indices.view(-1, k)
        
        output = torch.zeros_like(hidden_flat)
        
        # 处理每个专家
        for expert_idx, expert in enumerate(self.experts):
            # 找到需要该专家处理的位置
            expert_mask = (indices_flat == expert_idx)
            if expert_mask.any():
                # 获取对应的hidden states和权重
                for i in range(k):
                    mask = expert_mask[:, i]
                    if mask.any():
                        expert_output = expert(hidden_flat[mask])
                        weighted_output = expert_output * probs_flat[mask, i].unsqueeze(-1)
                        output[mask] += weighted_output
        
        return output.view(batch_size, seq_len, hidden_size)