import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def sample_gumbel(logits, eps=1e-20):
    U = torch.rand(logits.size()).to(logits)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits, eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
  if hard:
    shape = logits.size()
    _, k = y_soft.data.max(-1)
    y_hard = torch.zeros(*shape).to(y_soft)
    y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
    y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
  else:
    y = y_soft
  return y

class GWNET(BaseModel):
    '''
    Reference code: https://github.com/nnzhan/Graph-WaveNet
    '''
    def __init__(self, supports, adp_adj, dropout, residual_channels, dilation_channels, \
                 skip_channels, end_channels, expert_idx = None, kernel_size=2, blocks=4, layers=2, **args):
        super(GWNET, self).__init__(**args)
        self.supports = supports
        self.supports_len = len(supports)
        self.adp_adj = adp_adj
        self.expert_idx = expert_idx
        self.n_expert = len(expert_idx)
        self.register_buffer('mapping', torch.tensor(sum([[k] * len(v) for k, v in expert_idx.items()], [])))
        if adp_adj:
            dims = 10
            self.expert_embedding = nn.Parameter(torch.randn(self.n_expert, self.node_num, dims), requires_grad=True)
            self.ln1 = nn.Linear(self.horizon + 1 + self.node_num, self.n_expert)
            self.ln2 = nn.Linear(self.horizon + 1, dims)
            self.generator = gumbel_softmax
            self.supports_len += 1
        print('check supports length', len(supports), self.supports_len)
        
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1,kernel_size), dilation=new_dilation))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1,1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(GCN(dilation_channels, residual_channels, self.dropout, support_len=self.supports_len))
        self.receptive_field = receptive_field
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=self.output_dim * self.horizon,
                                    kernel_size=(1,1),
                                    bias=True)

    def dgconstruct(self, expert_embedding, input_embedding, ind):
        expert_adp = torch.einsum('bnd, bmd->bnm', input_embedding, expert_embedding[ind])
        remain_adp = torch.einsum('bnd, kmd->bknm', input_embedding, expert_embedding)
        return torch.sigmoid(expert_adp), torch.sigmoid(remain_adp)

    def forward(self, input, label=None):  # (b, t, n, f)
        input = input.transpose(1, 3)
        in_len = input.size(3)
        ind = (input[:, 1, 0, 0] * 288 // 6).long()
        ind = self.mapping[ind]
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        if self.adp_adj:
            signal_embed = self.ln2(x[:, 0])
            adp, remain_adp = self.dgconstruct(self.expert_embedding, signal_embed, ind)
            ft = torch.concat([x[:, 0].mean(-1), x[:, 0].mean(-2)], -1)
            pred = torch.tensor([]).to(ft)
            w = self.ln1(ft)
            for i in range(input.size(0)):
                w_i = w[i:i + 1]
                w_i = w_i[:, torch.arange(w_i.size(1)).to(w_i) != ind[i]]
                w_i = F.softmax(w_i, dim = 1)
                adp_i = remain_adp[i]
                adp_i = adp_i[torch.arange(len(adp_i)).to(adp_i) != ind[i]]
                pred = torch.concat([pred, torch.einsum('ai, ijk->ajk', w_i, adp_i.detach())])

            if self.training:
                graph = self.generator(adp, temperature=0.5, hard=True)
                loss = torch.abs(adp - pred).pow(2).mean()
                new_supports = self.supports + [graph]
            else:
                if self.expert == True:
                    graph = self.generator(adp, temperature=0.5, hard=True)
                    new_supports = self.supports + [graph]
                else:
                    graph = self.generator(pred, temperature=0.5, hard=False)
                    new_supports = self.supports + [graph]
        else:
            new_supports = self.supports

        x = self.start_conv(x)

        skip = 0
        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:         
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)
            
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
            
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        if self.training:
            return x, loss
        else:
            return x


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        try:
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
        except:
            x = torch.einsum('ncvl,nvw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)


    def forward(self,x):
        return self.mlp(x)

    
class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order


    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


