import torch
from torch import nn
from torch.nn import Parameter
from AE import AE
from opt import args
from GraphTransformer import GraphTransformerLayer
from GCNLayer import GCNLayer

class Tri_GFN(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,
                 gcn_n_enc_1, gcn_n_enc_2, gcn_n_enc_3,
                 gcn_n_dec_1, gcn_n_dec_2, gcn_n_dec_3,
                 graph_n_enc_1, graph_n_enc_2, graph_n_enc_3,
                 graph_n_dec_1, graph_n_dec_2, graph_n_dec_3,
                 n_input, n_z, n_clusters, v, n_node, device):
        super(Tri_GFN, self).__init__()
        self.ae = AE(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)

        ae_pre = 'ae_pretrain/{}.pkl'.format(args.name)
        self.ae.load_state_dict(torch.load(ae_pre, map_location='cpu'))
        print('Loading AE pretrain model:', ae_pre)

        self.graph_1 = GraphTransformerLayer(n_input, graph_n_enc_1)
        self.graph_2 = GraphTransformerLayer(graph_n_enc_1, graph_n_enc_2)
        self.graph_3 = GraphTransformerLayer(graph_n_enc_2, graph_n_enc_3)
        self.graph_4 = GraphTransformerLayer(graph_n_enc_3, n_z)
        self.graph_fc = GraphTransformerLayer(n_z, n_clusters)

        self.graph_5 = GraphTransformerLayer(n_z, graph_n_dec_1)
        self.graph_6 = GraphTransformerLayer(graph_n_dec_1, graph_n_dec_2)
        self.graph_7 = GraphTransformerLayer(graph_n_dec_2, graph_n_dec_3)
        self.graph_8 = GraphTransformerLayer(graph_n_dec_3, n_input)

        self.gcn_1 = GCNLayer(n_input, gcn_n_enc_1)
        self.gcn_2 = GCNLayer(gcn_n_enc_1, gcn_n_enc_2)
        self.gcn_3 = GCNLayer(gcn_n_enc_2, gcn_n_enc_3)
        self.gcn_4 = GCNLayer(gcn_n_enc_3, n_z)
        self.gcn_fc = GCNLayer(n_z, n_clusters)

        self.gcn_5 = GCNLayer(n_z, gcn_n_dec_1)
        self.gcn_6 = GCNLayer(gcn_n_dec_1, gcn_n_dec_2)
        self.gcn_7 = GCNLayer(gcn_n_dec_2, gcn_n_dec_3)
        self.gcn_8 = GCNLayer(gcn_n_dec_3, n_input)

        self.s = nn.Sigmoid()

        self.a = 0.5
        self.alpha = 0.45
        self.beta = 0.25
        self.gamma = 0.25

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.s = nn.Sigmoid()
        self.v = v
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, adj, edge_index):
        z_ae, x_bar, enc_h1, enc_h2, enc_h3 = self.ae(x)

        gcn_enc1 = self.gcn_1(x, edge_index, active=True)
        gcn_enc2 = self.gcn_2((1 - self.a) * gcn_enc1 + self.a * enc_h1, edge_index, active=True)
        gcn_enc3 = self.gcn_3((1 - self.a) * gcn_enc2 + self.a * enc_h2, edge_index, active=True)
        z_gcn = self.gcn_4((1 - self.a) * gcn_enc3 + self.a * enc_h3, edge_index, active=False)

        graph_enc1 = self.graph_1(x, edge_index, active=True)
        graph_enc2 = self.graph_2((1 - self.a) * graph_enc1 + self.a * enc_h1, edge_index, active=True)
        graph_enc3 = self.graph_3((1 - self.a) * graph_enc2 + self.a * enc_h2, edge_index, active=True)
        z_graph = self.gcn_4((1 - self.a) * graph_enc3 + self.a * enc_h3, edge_index, active=False)

        z_i = self.alpha * z_gcn + self.beta * z_ae + self.gamma * z_graph

        z_l = torch.spmm(adj, z_i)

        gcn_dec1 = self.gcn_5(z_gcn, edge_index, active=True)
        gcn_dec2 = self.gcn_6(gcn_dec1, edge_index, active=True)
        gcn_dec3 = self.gcn_7(gcn_dec2, edge_index, active=True)

        z_gcn_hat = self.gcn_8(gcn_dec3, edge_index, active=True)
        edge_gcn_hat = self.s(torch.mm(z_gcn, z_gcn.t()))

        graph_dec1 = self.graph_5(z_graph, edge_index, active=True)
        graph_dec2 = self.graph_6(graph_dec1, edge_index, active=True)
        graph_dec3 = self.graph_7(graph_dec2, edge_index, active=True)

        z_graph_hat = self.gcn_8(graph_dec3, edge_index, active=True)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z_l.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        return x_bar, z_gcn_hat, z_graph_hat, edge_gcn_hat, z_ae, q, q1, z_l
