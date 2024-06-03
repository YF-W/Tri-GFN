import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.cluster import KMeans
from AE import AE
from utils import LoadDataset, eva, get_data
from opt import args

def pretrain_ae(model, dataset, y, data_name):
    acc_reuslt = []
    nmi_result = []
    ari_result = []
    f1_result = []

    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(50):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()
            _, x_bar, _, _, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            z_ae, x_bar, _, _, _ = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z_ae.data.cpu().numpy())
            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)

            acc_reuslt.append(acc)
            nmi_result.append(nmi)
            ari_result.append(ari)
            f1_result.append(f1)

    torch.save(model.state_dict(), data_name + '.pkl')

print(args)
n_clusters = args.n_clusters
n_input = args.n_input
n_z = args.n_z

model = AE(
    ae_n_enc_1=500,
    ae_n_enc_2=500,
    ae_n_enc_3=2000,
    ae_n_dec_1=2000,
    ae_n_dec_2=500,
    ae_n_dec_3=500,
    n_input=n_input,
    n_z=n_z).cuda()

x, y, _, _ = get_data(args.name)
dataset = LoadDataset(x)
pretrain_ae(model, dataset, y, args.name)
