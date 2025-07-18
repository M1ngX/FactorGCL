import torch
import torch.nn as nn
from info_nce import InfoNCE
from torch_geometric.nn import HypergraphConv


class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(FeatureExtractor, self).__init__()
        self.normalize = nn.LayerNorm(input_size)
        self.linear = nn.Linear(input_size, input_size)
        self.leaky_relu = nn.LeakyReLU()  
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        # x: (num_stocks, seq_len, input_size)
        x = self.normalize(x)
        out = self.linear(x)
        out, _ = self.gru(x)
        e_s = out[:, -1, :]
        return e_s


class PriorBetaModule(nn.Module):
    def __init__(self, hidden_size):
        super(PriorBetaModule, self).__init__()
        self.hypergcn = HypergraphConv(hidden_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        
    def forward(self, e_s, industry_matrix):
        # x: (num_stocks, hidden_size)
        # industry_matrix: (num_stocks, num_industries)
        stock_indices, industry_indices = torch.where(industry_matrix == 1)
        hyperedge_index = torch.stack([stock_indices, industry_indices], dim=0)
        e_p = self.hypergcn(e_s, hyperedge_index)
        e_p = self.leaky_relu(e_p)
        return e_p
    

class SoftHypergraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, H):
        """
        x: (num_nodes, in_channels)
        H: (num_nodes, num_hyperedges), between 0 and 1
        """
        d_n, d_e = H.sum(dim=1), H.sum(dim=0) # degrees of nodes and hyperedges
        D_n_inv_sqrt = torch.diag(1.0 / torch.sqrt(d_n))
        D_n_inv_sqrt[D_n_inv_sqrt == float('inf')] = 0.0
        D_e_inv = torch.diag(1.0 / (d_e))
        D_e_inv[D_e_inv == float('inf')] = 0.0
        
        x_transformed = self.linear(x)  # (num_nodes, out_channels)
        out = D_n_inv_sqrt @ H @ D_e_inv @ H.T @ x_transformed  # (num_nodes, out_channels)
        return out


class HiddenBetaModule(nn.Module):
    def __init__(self, hidden_size, num_factors):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_factors, hidden_size), requires_grad=True)
        self.soft_hypergcn = SoftHypergraphConv(hidden_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU()
    
    def forward(self, e_r):
        # e_r: (num_stocks, hidden_size)
        beta_h = torch.sigmoid(e_r @ self.prototypes.T)
        # beta_h: (num_stocks, num_factors)
        e_h = self.soft_hypergcn(e_r, beta_h)
        e_h = self.leaky_relu(e_h)
        return e_h


class IndividualAlphaModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU()
    
    def forward(self, e_residual):
        # e_residual: (num_stocks, hidden_size)
        e_alpha = self.linear(e_residual)
        e_alpha = self.leaky_relu(e_alpha)
        return e_alpha


class _FactorGCL(nn.Module):
    def __init__(self, input_size, hidden_size, num_factors, num_layers, dropout):
        super(_FactorGCL, self).__init__()
        self.feature_extractor = FeatureExtractor(input_size, hidden_size, num_layers, dropout)
        self.prior_beta_module = PriorBetaModule(hidden_size)
        self.hidden_beta_module = HiddenBetaModule(hidden_size, num_factors)
        self.individual_alpha_module = IndividualAlphaModule(hidden_size)
        self.fc = nn.Linear(hidden_size * 3, 1)
        
    def forward(self, x, industry_matrix):
        e_s = self.feature_extractor(x)
        e_p = self.prior_beta_module(e_s, industry_matrix)
        e_r = e_s - e_p
        e_h = self.hidden_beta_module(e_r)
        e_residual = e_r - e_h
        e_alpha = self.individual_alpha_module(e_residual)

        out = torch.concat([e_p, e_h, e_alpha], dim=1)
        out = self.fc(out)
        out = out.reshape(-1).squeeze()
        return out
    
    def predict(self, x, industry_matrix):
        self.eval()
        with torch.no_grad():
            return self.forward(x, industry_matrix)


class FactorGCL(nn.Module):
    def __init__(self, input_size, hidden_size, num_factors, num_layers, dropout, criterion=nn.MSELoss(), gamma=0.1):
        super(FactorGCL, self).__init__()
        self.feature_extractor = FeatureExtractor(input_size, hidden_size, num_layers, dropout)
        self.prior_beta_module = PriorBetaModule(hidden_size)
        self.hidden_beta_module = HiddenBetaModule(hidden_size, num_factors)
        self.individual_alpha_module = IndividualAlphaModule(hidden_size)
        self.fc = nn.Linear(hidden_size * 3, 1)

        # self.negative_mode = 'paired'
        # using 'paired' mode for negative sampling will make the traning much slower
        self.negative_mode = None
        self.feature_extractor_future = FeatureExtractor(input_size, hidden_size, num_layers, dropout)
        self.info_nce = InfoNCE(negative_mode=self.negative_mode)
        self.criterion = criterion
        self.gamma = gamma

    def forward(self, x, x_future, y, industry_matrix):
        e_s = self.feature_extractor(x)
        e_p = self.prior_beta_module(e_s, industry_matrix)
        e_r = e_s - e_p
        e_h = self.hidden_beta_module(e_r)
        e_residual = e_r - e_h
        e_alpha = self.individual_alpha_module(e_residual)

        y_pred = torch.concat([e_p, e_h, e_alpha], dim=1)
        y_pred = self.fc(y_pred).reshape(-1)
        pred_loss = self.criterion(y_pred, y)

        # Temporal Residual Contrastive Learning
        e_s_future = self.feature_extractor_future(x_future)
        e_p_future = self.prior_beta_module(e_s_future, industry_matrix)
        e_r_future = e_s_future - e_p_future
        e_h_future = self.hidden_beta_module(e_r_future)
        # e_residual_future = e_r_future - e_h_future
        # e_alpha_future = self.individual_alpha_module(e_residual_future)
        e_alpha_future = e_r_future - e_h_future

        if self.negative_mode == 'paired':
            # negative_keys: (num_stocks, num_stocks - 1，hidden_size)
            # negative_keys are the future representations of all stocks except the current one
            mask = ~torch.eye(x.size(0), dtype=torch.bool)
            expanded = e_alpha_future.unsqueeze(0).expand(x.size(0), -1, -1)
            negative_keys = expanded[mask].view(x.size(0), x.size(0) - 1, -1)
            info_nce_loss = self.info_nce(e_alpha, e_alpha_future, negative_keys)
        elif self.negative_mode is None:
            info_nce_loss = self.info_nce(e_alpha, e_alpha_future)
        
        loss = pred_loss + self.gamma * info_nce_loss
        return loss
    
    def predict(self, x, industry_matrix):
        self.eval()
        with torch.no_grad():
            e_s = self.feature_extractor(x)
            e_p = self.prior_beta_module(e_s, industry_matrix)
            e_r = e_s - e_p
            e_h = self.hidden_beta_module(e_r)
            e_residual = e_r - e_h
            e_alpha = self.individual_alpha_module(e_residual)

            out = torch.concat([e_p, e_h, e_alpha], dim=1)
            out = self.fc(out)
            out = out.reshape(-1).squeeze()
            return out


if __name__ == "__main__":
    # testing
    X = torch.randn(3000, 20, 64)  # (num_stocks, seq_len, input_size)
    X_future = torch.randn(3000, 10, 64)  # (num_stocks, seq_len_future, input_size)
    industry_matrix = torch.randint(0, 2, (3000, 50))  # (num_stocks, num_industries)
    y = torch.randn(3000)  # (num_stocks)

    input_size = 64
    hidden_size = 32
    num_factors = 128
    num_layers = 2
    dropout = 0.1
    criterion = nn.MSELoss()
    gamma = 0.1

    model1 = _FactorGCL(input_size, hidden_size, num_factors, num_layers, dropout)
    model2 = FactorGCL(input_size, hidden_size, num_factors, num_layers, dropout, criterion, gamma)
    pred = model1(X, industry_matrix)
    print(pred.shape)  # should be (3000,)
    loss = model2(X, X_future, y, industry_matrix)
    print(loss)  # should be a scalar tensor