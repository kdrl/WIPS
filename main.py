import sys
import argparse
import logging
import random
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import models #, train, data # will be updated soon

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-debug', help='Print debug output', action='store_true', default=True)
    parser.add_argument('-save_dir', help='Path for saving checkpoints', type=str, required=True)
    parser.add_argument('-exp_name', help='Experiment name', type=str, default=str(datetime.datetime.now()).replace(" ","_"))
    parser.add_argument('-graph_type', help='Graph type: "webkb", "collab" or "hirearchy"', type=str, required=True)
    parser.add_argument('-nproc', help='Number of processes', type=int, default=1)
    parser.add_argument('-ndproc', help='Number of data loading processes', type=int, default=4)
    parser.add_argument('-neproc', help='Number of eval processes', type=int, default=8)
    parser.add_argument('-seed', help='Random seed', type=int, required=False)

    parser.add_argument('-dblp_path', help='dblp_path', type=str, required=False)
    parser.add_argument('-webkb_path', help='webkb_path', type=str, required=False)
    parser.add_argument('-word2vec_path', help='word2vec_path', type=str, required=False)

    parser.add_argument('-iter', help='Number of iterations', type=int, default=10)
    parser.add_argument('-eval_each', help='Run evaluation at every n-th iter', type=int, default=1)
    parser.add_argument('-init_lr', help='Initial learning rate', type=float, default=0.1)
    parser.add_argument('-batchsize', help='Batchsize', type=int, default=2)
    parser.add_argument('-negs', help='Number of negative samples', type=int, default=10)
    parser.add_argument('-smoothing_rate_for_node', help='Smoothing rate for the negative sampling', type=float, default=1.0)

    parser.add_argument('-hidden_layer_num', help='Number of hidden layers', type=int, default=2)
    parser.add_argument('-hidden_size', help='Number of units in a hidden layer', type=int, default=2000)

    parser.add_argument('-model_name', help='Model: "IPS", "SIPS", "NPD", "IPDS" or "WIPS"', type=str, required=True)
    parser.add_argument('-task', help='', type=str, default="reconst")
    parser.add_argument('-parameter_num', help='Parameter number K for each node', type=int, default=100)
    parser.add_argument('-neg_ratio', help='Dimension ratio for negative IPS in IPDS', type=float, default=0.0, required=False)
    parser.add_argument('-neg_dim', help='Dimension for negative IPS in IPDS', type=int, default=0, required=False)
    parser.add_argument('-wips_positive_initialization', help='', type=int, default=0, required=False)

    opt = parser.parse_args()

    if opt.seed == None: opt.seed = random.randint(1, 1000000)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    opt.cuda = torch.cuda.is_available()
    if opt.cuda: torch.cuda.manual_seed(opt.seed)
    torch.set_default_tensor_type('torch.FloatTensor')
    opt.exp_name += "@"+str(datetime.datetime.now()).replace(" ","_")
    if opt.debug: log_level = logging.DEBUG
    else: log_level = logging.INFO
    log = logging.getLogger(opt.exp_name)
    fileHandler = logging.FileHandler(f'{opt.save_dir}/{opt.exp_name}.log')
    streamHandler = logging.StreamHandler()
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    log.setLevel(log_level)
    log.info(f"Experiment {opt.exp_name} start at {sys.version} with setting:\n{str(opt.__dict__)}")

    if opt.graph_type == "hirearchy":
        word2id, id2freq, edges2freq, vectors = data.preprocess_hirearchy(opt.word2vec_path, use_rich_information=False)
    elif opt.graph_type == "collab":
        word2id, id2freq, edges2freq, vectors = data.preprocess_co_author_network(opt.dblp_path)
    elif opt.graph_type == "webkb":
        word2id, id2freq, edges2freq, vectors = data.preprocess_webkb_network(opt.webkb_path)
    else:
        raise Exception(f"{opt.graph_type} is not supported")
    dataset = data.GraphDataset(word2id, id2freq, edges2freq, opt.negs, opt.smoothing_rate_for_node, vectors, opt.task, opt.seed)
    opt.data_vectors = dataset.data_vectors
    opt.total_node_num = dataset.total_node_num
    opt.train_node_num = dataset.train_node_num

    if opt.model_name == "IPDS":
        if opt.neg_dim == 0:
          opt.neg_dim = np.round(opt.neg_ratio * opt.parameter_num)
        if opt.neg_dim == 0 : opt.neg_dim = 1
        if opt.neg_dim == opt.parameter_num : opt.neg_dim = opt.parameter_num-1
        opt.neg_dim = int(opt.neg_dim)

    if opt.model_name == "WIPS":
        opt.wips_positive_initialization = 1.0/opt.parameter_num


    model = getattr(models, opt.model_name)(opt.__dict__)

    filtered_parameters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            filtered_parameters.append(param)
    params_num = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])

    if opt.cuda:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = Adam(
        filtered_parameters,
        lr=opt.init_lr
    )

    train.trainer(model, dataset, lossfn, optimizer, opt.__dict__, log, opt.cuda)

def lossfn(preds, target):
    pos_score = preds.narrow(1, 0, 1)
    neg_score = preds.narrow(1, 1, preds.size(1) - 1)
    pos_loss = F.logsigmoid(pos_score).squeeze().sum()
    neg_loss = F.logsigmoid(-1*neg_score).squeeze().sum()
    loss = pos_loss + neg_loss
    return -1*loss

if __name__ == '__main__':
    main()
