from collections import OrderedDict
import torch
import torch.nn.init as init
from torch import nn
from torch.autograd import Function


class Embedding(nn.Module):

    def __init__(self, opt):
        super(Embedding, self).__init__()
        self.wips_positive_initialization = opt["wips_positive_initialization"]
        self.unified = True
        self.model = opt["model_name"]
        self.total_node_num = opt["total_node_num"]
        self.train_node_num = opt["train_node_num"]
        self.parameter_num = opt["parameter_num"]
        assert opt["data_vectors"].shape[0] == opt["total_node_num"]
        assert opt["hidden_layer_num"] >= 1
        self.data_vectors = nn.Embedding(
            opt["data_vectors"].shape[0], opt["data_vectors"].shape[1]
        )
        self.data_vectors.weight.data = torch.from_numpy(opt["data_vectors"]).float()
        self.data_vectors.weight.requires_grad = False
        self.U_NN = self.build_NN(opt["data_vectors"],opt["hidden_layer_num"],opt["hidden_size"])
        self.U = nn.Linear(opt["hidden_size"], opt["parameter_num"])

    def initialization(self):
        print(f"Parameter list : {self.named_parameters()}")

        for name, param in self.named_parameters():
            if 'data_vectors' in name:
                print("Skip init. {} since it is already given".format(name))
                continue
            if 'ips_weight' in name:
                assert self.wips_positive_initialization:
                print("Init. {} with uniform distribution 0.0 - {}".format(name, self.wips_positive_initialization))
                init.uniform_(param, 0.0, self.wips_positive_initialization)
                print(f" -> param : {param.data.numpy()}")
                continue
            if 'bias' in name:
                print("Init. {} to zero".format(name))
                init.constant_(param, 0.0)
            elif 'weight' in name:
                print("Init. {} with He".format(name))
                init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
            else:
                raise Exception(name)

    def build_NN(self, given_data_vectors, hidden_layer_num, hidden_size):
        NN = [("fc0", nn.Linear(given_data_vectors.shape[1], hidden_size))]
        for i in range(1, hidden_layer_num):
            NN.extend([(f"relu{i-1}", nn.ReLU(True)), (f"fc{i}", nn.Linear(hidden_size, hidden_size))])
        NN.append((f"relu{hidden_layer_num-1}", nn.ReLU(True)))
        return nn.Sequential(OrderedDict(NN))

    def forward(self, inputs):
        inputs = self.U_NN(self.data_vectors(inputs))
        e = self.U(inputs)
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        dists = self.distfn([s, o]).squeeze(-1)
        return -dists

    def embed(self):
        embeddings = self.U(self.U_NN(self.data_vectors.state_dict()['weight'])).data.cpu().numpy()
        return [embeddings]

    def get_similarity(self, inputs):
        return self.forward(inputs)


class WIPS(Embedding):
    """ Weighted Inner Product Similarity (WIPS)"""

    def __init__(self, opt):
        super(WIPS, self).__init__(opt)
        self.ips_weight = nn.Parameter(torch.zeros(opt["parameter_num"]))
        self.initialization()

    def distfn(self, input, w=None):
        u, v = input
        if w is None: w=self.ips_weight
        return -torch.sum(u * v * w, dim=-1)

    def get_ips_weight(self):
        return self.ips_weight.data.cpu().numpy()


class IPS(Embedding):
    """ Inner Product Similarity (IPS)"""

    def __init__(self, opt):
        super(IPS, self).__init__(opt)
        self.initialization()

    def distfn(self, input):
        u, v = input
        return -(torch.sum(u * v, dim=-1))


class SIPS(Embedding):
    """ Shifted Inner Product Similarity (SIPS)"""

    def __init__(self, opt):
        super(SIPS, self).__init__(opt)
        self.U = nn.Embedding(
            opt["train_node_num"],
            opt["parameter_num"]-1,
        )
        self.U_bias = nn.Embedding(
            opt["train_node_num"],
            1
        )
        self.U = nn.Linear(opt["hidden_size"], opt["parameter_num"]-1)
        self.U_bias = nn.Linear(opt["hidden_size"], 1)
        self.initialization()

    def distfn(self, input):
        u, u_bias, v, v_bias = input
        return -(torch.sum(u * v, dim=-1) + u_bias.squeeze(-1) + v_bias.squeeze(-1))

    def forward(self, inputs):
        inputs = self.U_NN(self.data_vectors(inputs))
        e = self.U(inputs)
        eb = self.U_bias(inputs)
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        ob = eb.narrow(1, 1, e.size(1) - 1)
        sb = eb.narrow(1, 0, 1).expand_as(ob)
        dists = self.distfn([s, sb, o, ob]).squeeze(-1)
        return -dists

    def embed(self):
        tmp = self.U_NN(self.data_vectors.state_dict()['weight'])
        return [self.U(tmp).data.cpu().numpy(), self.U_bias(tmp).data.cpu().numpy()]


class IPDS(Embedding):
    """ Inner Product Difference Similarity (IPDS)"""

    def __init__(self, opt):
        super(IPDS, self).__init__(opt)
        if opt["data_vectors"] is None:
            self.U = nn.Embedding(
                opt["train_node_num"],
                opt["parameter_num"]-opt["neg_dim"]
            )
            self.U_neg = nn.Embedding(
                opt["train_node_num"],
                opt["neg_dim"]
            )
        else:
            self.U = nn.Linear(opt["hidden_size"], opt["parameter_num"]-opt["neg_dim"])
            self.U_neg = nn.Linear(opt["hidden_size"], opt["neg_dim"])
        self.initialization()

    def distfn(self, input):
        u, u_neg, v, v_neg = input
        return -(torch.sum(u * v, dim=-1) - torch.sum(u_neg * v_neg, dim=-1))

    def forward(self, inputs):
        inputs = self.U_NN(self.data_vectors(inputs))
        u = self.U(inputs)
        u_neg = self.U_neg(inputs)
        ui = u.narrow(1, 1, u.size(1) - 1)
        uj = u.narrow(1, 0, 1).expand_as(ui)
        u_negi = u_neg.narrow(1, 1, u_neg.size(1) - 1)
        u_negj = u_neg.narrow(1, 0, 1).expand_as(u_negi)
        dists = self.distfn([ui, u_negi, uj, u_negj]).squeeze(-1)
        return -dists

    def embed(self):
        tmp = self.U_NN(self.data_vectors.state_dict()['weight'])
        return [self.U(tmp).data.cpu().numpy(), self.U_neg(tmp).data.cpu().numpy()]


class NPD(Embedding):
    """ Negative Poincaré Distance
    Based on the implementation of Poincaré Embedding : https://github.com/facebookresearch/poincare-embeddings
    """

    eps = 1e-5

    def __init__(self, opt):
        super(NPD, self).__init__(opt)
        self.dist = PDF
        if opt["data_vectors"] is None:
            self.U = nn.Embedding(
                opt["train_node_num"],
                opt["parameter_num"],
                max_norm=1
            )
        else:
            self.U = nn.Linear(opt["hidden_size"], opt["parameter_num"])
        self.initialization()

    def distfn(self, input):
        s, o = input
        return self.dist()(s, o)

    def forward(self, inputs):
        e = self.U(self.U_NN(self.data_vectors(inputs)))
        n = torch.norm(e, p=2, dim=2)
        mask = (n >= 1.0)
        f = n * mask.type(n.type())
        f[f!=0] /= (1.0-eps)
        f[f==0] = 1.0
        e = e.clone()/f.unsqueeze(2)
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        dists = self.distfn([s, o]).squeeze(-1)
        return -dists

    def embed(self):
        e = self.U(self.U_NN(self.data_vectors.state_dict()['weight']))
        n = torch.norm(e, p=2, dim=1)
        mask = (n >= 1.0)
        f = n * mask.type(n.type())
        f[f!=0] /= (1.0-eps)
        f[f==0] = 1.0
        e = e.clone()/f.unsqueeze(1)
        return [e.data.cpu().numpy()]


class PDF(Function):
    """ Poincaré Distance Function
    Based on the implementation of Poincaré Embedding : https://github.com/facebookresearch/poincare-embeddings
    """

    eps = 1e-5

    def grad(self, x, v, sqnormx, sqnormv, sqdist):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) / torch.pow(alpha, 2)).unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = torch.sqrt(torch.pow(z, 2) - 1)
        z = torch.clamp(z * beta, min=self.eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    def forward(self, u, v):
        self.save_for_backward(u, v)
        self.squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, 1 - self.eps)
        self.sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, 1 - self.eps)
        self.sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        x = self.sqdist / ((1 - self.squnorm) * (1 - self.sqvnorm)) * 2 + 1
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    def backward(self, g):
        u, v = self.saved_tensors
        g = g.unsqueeze(-1)
        gu = self.grad(u, v, self.squnorm, self.sqvnorm, self.sqdist)
        gv = self.grad(v, u, self.sqvnorm, self.squnorm, self.sqdist)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv
