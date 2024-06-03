import argparse

parser = argparse.ArgumentParser(description='TriGFN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default="acm")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--n_z', type=int, default=10)
parser.add_argument('--alpha', type=int, default=0.1)
parser.add_argument('--beta', type=int, default=0.08)

args = parser.parse_args()
print("Network setting…")

if args.name == 'acm':
    args.lr = 5e-5
    args.n_clusters = 3
    args.n_input = 1870
elif args.name == 'dblp':
    args.lr = 2e-3
    args.n_clusters = 4
    args.n_input = 334
elif args.name == 'cite':
    args.lr = 4e-5
    args.n_clusters = 6
    args.n_input = 3703
elif args.name == 'phy':
    args.lr = 1e-4
    args.n_clusters = 5
    args.n_input = 8415
    args.cuda = False
elif args.name == 'cora':
    args.lr = 1e-4
    args.n_clusters = 7
    args.n_input = 1433
elif args.name == 'pubmed':
    args.lr = 1e-4
    args.n_clusters = 3
    args.n_input = 500
elif args.name == 'hhar':
    args.lr = 1e-4
    args.n_clusters = 6
    args.n_input = 561
elif args.name == 'reut':
    args.lr = 1e-4
    args.n_clusters = 4
    args.n_input = 2000
elif args.name == 'usps':
    args.lr = 1e-3
    args.n_clusters = 10
    args.n_input = 256

else:
    print("error!")
