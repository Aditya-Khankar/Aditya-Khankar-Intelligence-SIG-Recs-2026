import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    """
    LR baseline: Logistic regression on numerics + 1-d embedding per categorical field.
    Provides a linear reference to evaluate the benefit of nonlinearity.
    """
    def __init__(self, num_numeric, cat_cardinalities):
        super().__init__()
        self.num_numeric = num_numeric
        self.numeric_layer = nn.Linear(num_numeric, 1)
        
        # 1-d embedding per categorical field serves as logistic regression weight
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, 1) for card in cat_cardinalities
        ])
        
    def forward(self, x_num, x_cat):
        # Numeric linear part
        out = self.numeric_layer(x_num)
        
        # Categorical part: sum of 1-d embeddings
        for i, emb in enumerate(self.cat_embeddings):
            # x_cat[:, i] has shape [B], emb() gives [B, 1]
            out = out + emb(x_cat[:, i])
            
        return out.squeeze(1)


class BaseMLP(nn.Module):
    """
    Base MLP for Neural CTR.
    Concatenates numerics and categorical embeddings, then passes through an MLP.
    """
    def __init__(self, num_numeric, cat_cardinalities, embed_dim, hidden_layers, dropout_rate=0.0):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in cat_cardinalities
        ])
        
        input_dim = num_numeric + len(cat_cardinalities) * embed_dim
        
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x_num, x_cat):
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        # Concatenate numerics with all categorical embeddings flattened
        x_concat = torch.cat([x_num] + cat_embs, dim=1)
        out = self.mlp(x_concat)
        return out.squeeze(1)


class MLPA(BaseMLP):
    def __init__(self, num_numeric, cat_cardinalities, embed_dim=16):
        super().__init__(num_numeric, cat_cardinalities, embed_dim, [128], dropout_rate=0.0)

class MLPB(BaseMLP):
    def __init__(self, num_numeric, cat_cardinalities, embed_dim=16):
        super().__init__(num_numeric, cat_cardinalities, embed_dim, [256, 256], dropout_rate=0.2)

class MLPC(BaseMLP):
    def __init__(self, num_numeric, cat_cardinalities, embed_dim=16):
        super().__init__(num_numeric, cat_cardinalities, embed_dim, [512, 512, 512], dropout_rate=0.3)


class DCN(nn.Module):
    """
    Deep & Cross Network (DCN).
    Parallel cross network (3 layers) and deep MLP, concatenated for final logit.
    """
    def __init__(self, num_numeric, cat_cardinalities, embed_dim=16, mlp_hidden=[256, 256], mlp_dropout=0.2, cross_layers=3):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in cat_cardinalities
        ])
        
        self.input_dim = num_numeric + len(cat_cardinalities) * embed_dim
        
        # Deep network
        layers = []
        curr_dim = self.input_dim
        for hidden_dim in mlp_hidden:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            if mlp_dropout > 0:
                layers.append(nn.Dropout(mlp_dropout))
            curr_dim = hidden_dim
        self.deep = nn.Sequential(*layers)
        
        # Cross network parameters
        self.cross_layers = cross_layers
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.input_dim)) for _ in range(cross_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.input_dim)) for _ in range(cross_layers)
        ])
        
        # Initialize cross parameters
        for w, b in zip(self.cross_weights, self.cross_biases):
            nn.init.normal_(w, mean=0.0, std=0.01)
            nn.init.zeros_(b)
            
        self.final_linear = nn.Linear(curr_dim + self.input_dim, 1)

    def forward(self, x_num, x_cat):
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x0 = torch.cat([x_num] + cat_embs, dim=1)  # Shape: [B, input_dim]
        
        # Deep Network
        deep_out = self.deep(x0)
        
        # Cross Network
        xl = x0
        for i in range(self.cross_layers):
            # Efficiently compute x0 * (xl @ w) + b + xl
            xl_w = torch.matmul(xl, self.cross_weights[i])  # Shape: [B]
            cross_term = x0 * xl_w.unsqueeze(1)             # Shape: [B, input_dim]
            xl = cross_term + self.cross_biases[i] + xl
            
        # Concatenate and final logit
        out = self.final_linear(torch.cat([deep_out, xl], dim=1))
        return out.squeeze(1)
