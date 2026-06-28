"""
手写 LayerNorm 演示。

LayerNorm 的核心思想：
    对每一个样本内部的最后若干维做归一化，而不是像 BatchNorm 那样跨 batch 统计。

在 Transformer 中，常见输入形状是：
    x: [batch_size, seq_len, hidden_size]
        [1,1,512]

如果 normalized_shape=hidden_size，那么 LayerNorm 会对每个 token 的 hidden_size 维度
单独计算均值和方差：
    mean: [batch_size, seq_len, 1]
    var:  [batch_size, seq_len, 1]

也就是说：
    batch 中不同句子之间不会互相影响；
    同一句子中不同 token 之间也不会互相影响；
    每个 token 只在自己的 hidden_size 维度上做归一化。
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    手写版 LayerNorm。

    公式：
        y = gamma * (x - mean) / sqrt(var + eps) + beta

    参数：
        normalized_shape:
            要归一化的最后一维大小。
            在 Transformer 里通常就是 hidden_size / embedding_size。
        eps:
            防止除以 0 的小常数。
        elementwise_affine:
            是否使用可学习的 gamma 和 beta。
            PyTorch 的 nn.LayerNorm 默认开启。
    """

    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            # gamma 是缩放参数，初始为 1。
            # 它允许模型在归一化后重新调整每个 hidden 维度的幅度。
            self.gamma = nn.Parameter(torch.ones(normalized_shape))

            # beta 是平移参数，初始为 0。
            # 它允许模型在归一化后重新调整每个 hidden 维度的位置。
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            # 如果不使用可学习仿射参数，就把 gamma / beta 注册为 None。
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x: torch.Tensor):
        """
        前向传播。

        输入：
            x: [..., normalized_shape]

        输出：
            y: 和 x 形状相同。
        """

        # 只沿最后一维计算均值。
        # keepdim=True 是为了保留最后一维，方便后面和 x 自动广播。
        mean = x.mean(dim=-1, keepdim=True)

        # LayerNorm 使用的是总体方差 unbiased=False。
        # 这和 torch.nn.LayerNorm 的实现保持一致。
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 标准化后，每个 token 的最后一维均值接近 0，方差接近 1。
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            # gamma / beta 的形状是 [hidden_size]。
            # 当 x_norm 是 [B, L, hidden_size] 时，PyTorch 会自动广播到前面的 B 和 L。
            x_norm = x_norm * self.gamma + self.beta

        return x_norm


def main():
    """
    最小验证：
    1. 构造一个模拟 Transformer hidden states 的张量；
    2. 创建手写 LayerNorm 和 PyTorch LayerNorm；
    3. 把 PyTorch LayerNorm 的参数复制给手写 LayerNorm；
    4. 比较两者输出是否几乎一致。
    """

    torch.manual_seed(42)

    batch_size = 2
    seq_len = 3
    hidden_size = 4

    # 模拟 Transformer 中某一层的 hidden states。
    x = torch.randn(batch_size, seq_len, hidden_size)

    manual_ln = LayerNorm(hidden_size)
    torch_ln = nn.LayerNorm(hidden_size)

    # 为了公平比较，把 PyTorch LayerNorm 的 weight / bias 复制给手写版本。
    # nn.LayerNorm 里 weight 对应 gamma，bias 对应 beta。
    with torch.no_grad():
        manual_ln.gamma.copy_(torch_ln.weight)
        manual_ln.beta.copy_(torch_ln.bias)

    manual_out = manual_ln(x)
    torch_out = torch_ln(x)

    print("输入 x shape:", x.shape)
    print("手写 LayerNorm 输出 shape:", manual_out.shape)
    print("PyTorch LayerNorm 输出 shape:", torch_out.shape)

    # 如果实现正确，最大绝对误差应该非常小，一般在 1e-6 量级。
    max_diff = (manual_out - torch_out).abs().max()
    print("手写实现和 nn.LayerNorm 的最大误差:", max_diff.item())

    # 观察归一化效果：对最后一维计算均值和方差。
    # 开启 gamma/beta 且初始参数为 gamma=1, beta=0 时，
    # 输出最后一维的均值应接近 0，方差应接近 1。
    print("输出最后一维均值:")
    print(manual_out.mean(dim=-1))

    print("输出最后一维方差:")
    print(manual_out.var(dim=-1, unbiased=False))


if __name__ == "__main__":
    main()
