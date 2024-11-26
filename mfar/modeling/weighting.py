import torch

class LinearWeights(torch.nn.Module):
    """
    Linear weights with two options:
    1) Regular mixture of fields
    2) Query conditioning over weights
    """

    def __init__(self, emb_size, num_fields, query_cond=False):
        super(LinearWeights, self).__init__()
        self.query_cond = query_cond
        # emb_size = 1 if no query conditioning
        self.weight = torch.nn.Parameter(torch.ones(emb_size, num_fields),
                                         requires_grad=True)

    def forward(self,
                x,   # [Batch, Samples, Field]
                q,   # Optional([Batch, Emb])
        ) -> torch.Tensor: # [Batch, Samples]
        """
        The batch size of x and q must be the same or else this should fail, or the math will be wrong.
        """
        if self.query_cond:
            weights = q @ self.weight # [Batch, Emb] * [Emb, Field] -> [Batch, Field]
        else:
            weights = self.weight.transpose(1, 0) # [1, Field]
        weights_dist = torch.softmax(weights, dim=1) # [ ?, Field]
        return torch.sum(weights_dist.unsqueeze(1) * x, dim=-1)
