import torch
from torch.optim import AdamW
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F
from Tri_GFN import Tri_GFN
from utils import setup_seed, target_distribution, eva, LoadDataset, get_data
from opt import args
import datetime
from logger import Logger, metrics_info, record_info

nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')

def train(model, x, y):
    acc_reuslt = []
    nmi_result = []
    ari_result = []
    f1_result = []
    original_acc = -1
    metrics = [' acc', ' nmi', ' ari', ' f1']
    logger = Logger(args.name + '==' + nowtime)
    logger.info(model)
    logger.info(args)
    logger.info(metrics_info(metrics))

    n_clusters = args.n_clusters
    optimizer = AdamW(model.parameters(), lr=args.lr)
    with torch.no_grad():
        z, _, _, _, _ = model.ae(x)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    cluster_id = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    logger.info("%s%s" % ('Initialization: ',record_info(eva(y, cluster_id))))

    for epoch in range(200):
        x_bar, z_gcn_hat, z_graph_hat, edge_gcn_hat, z_ae, q, q1, z_l = model(x, adj, edge_index)
        if epoch % 1 == 0:
            tmp_q = q.data
            p = target_distribution(tmp_q)

        ae_loss = F.mse_loss(x_bar, x)
        loss_w = F.mse_loss(0.5 * z_gcn_hat + 0.5 * z_graph_hat, torch.spmm(adj, x))
        loss_a = F.mse_loss(edge_gcn_hat, adj.to_dense())
        loss_gae = loss_w + 0.1 * loss_a
        re_loss = 1 * ae_loss + 1 * loss_gae
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        q1q_loss = F.kl_div(q1.log(), q, reduction='batchmean')
        loss = 1 * re_loss + args.alpha * kl_loss + args.beta * q1q_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        res = q1.data.cpu().numpy().argmax(1) #Q1

        acc, nmi, ari, f1 = eva(y, res, str(epoch) + 'Q1')

        logger.info("epoch%d%s:\t%s" % (epoch, ' Q1', record_info([acc, nmi, ari, f1])))

        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        f1_result.append(f1)

        if acc >= original_acc:
            original_acc = acc
            torch.save(model.state_dict(), './model_save/{}.pkl'.format(args.name))

    best_acc = max(acc_reuslt)
    t_nmi = nmi_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]
    t_ari = ari_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]
    t_f1 = f1_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]
    t_epoch = np.where(acc_reuslt == np.max(acc_reuslt))[0][0]
    logger.info("%sepoch%d:\t%s" % ('Best Acc is at ', t_epoch, record_info([best_acc, t_nmi, t_ari, t_f1])))


if __name__ == "__main__":

    setup_seed(2018)
    print(args)
    device = torch.device("cuda")
    x, y, adj, edge_index = get_data(args.name)
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
    adj = torch.tensor(adj, dtype=torch.float32, device=device)
    edge_index = edge_index.t().contiguous()
    dataset = LoadDataset(x)
    x = torch.Tensor(dataset.x).to(device)
    model = Tri_GFN(
        ae_n_enc_1=500,
        ae_n_enc_2=500,
        ae_n_enc_3=2000,
        ae_n_dec_1=2000,
        ae_n_dec_2=500,
        ae_n_dec_3=500,
        gcn_n_enc_1=500,
        gcn_n_enc_2=500,
        gcn_n_enc_3=2000,
        gcn_n_dec_1=2000,
        gcn_n_dec_2=500,
        gcn_n_dec_3=500,
        graph_n_enc_1=500,
        graph_n_enc_2=500,
        graph_n_enc_3=2000,
        graph_n_dec_1=2000,
        graph_n_dec_2=500,
        graph_n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        v=1.0,
        n_node=x.size()[0],
        device=device,).to(device)

    train(model, x, y)
