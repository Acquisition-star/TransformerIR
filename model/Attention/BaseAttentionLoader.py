class BaseAttentionLoader:
    def __init__(self, index, iter, dim, bias, attn_config):
        self.index = index
        self.iter = iter
        self.dim = dim
        self.bias = bias
        self.attn_config = attn_config

    def load(self):
        raise NotImplementedError
