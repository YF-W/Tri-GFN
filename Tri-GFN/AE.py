from torch import nn
from torch.nn import Linear

class AE(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, ae_n_enc_1)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)
        self.enc_3 = Linear(ae_n_enc_2, ae_n_enc_3)
        self.z_layer = Linear(ae_n_enc_3, n_z)

        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.dec_3 = Linear(ae_n_dec_2, ae_n_dec_3)
        self.x_bar_layer = Linear(ae_n_dec_3, n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        enc_h1 = self.act(self.enc_1(x))
        enc_h2 = self.act(self.enc_2(enc_h1))
        enc_h3 = self.act(self.enc_3(enc_h2))
        z_ae = self.z_layer(enc_h3)

        dec_h1 = self.act(self.dec_1(z_ae))
        dec_h2 = self.act(self.dec_2(dec_h1))
        dec_h3 = self.act(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return z_ae, x_bar, enc_h1, enc_h2, enc_h3
