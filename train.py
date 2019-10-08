import os
import sys
import timeit
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp
from torch.utils.data import DataLoader


def trainer(model, dataset, lossfn, optimizer, opt, log, cuda):
    loader = DataLoader(
        dataset,
        batch_size=opt["batchsize"],
        collate_fn=dataset.collate,
        sampler=dataset.sampler
    )

    max_ROCAUC = (-1, -1)
    max_ROCAUC_model = None
    max_ROCAUC_model_on_test = None

    iter_counter = 0
    former_loss = np.Inf

    t_start = timeit.default_timer()

    assert opt["iter"] % opt["eval_each"] == 0
    pbar = tqdm(total=opt["eval_each"])

    while True:
        train_loss = []
        loss = None

        for inputs, targets in loader:
            pbar.update(1)

            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            optimizer.zero_grad()
            preds = model(inputs)
            loss = lossfn(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            iter_counter+=1

            if iter_counter % opt["eval_each"] == 0:
                pbar.close()
                model.eval()
                ROCAUC_train, ROCAUC_valid, ROCAUC_test, eval_elapsed = evaluation(model, dataset.neighbor_train, dataset.neighbor_valid, dataset.neighbor_test, dataset.task, log, opt["neproc"], cuda, True)
                model.train()
                if ROCAUC_valid > max_ROCAUC[0]:
                    max_ROCAUC = (ROCAUC_valid, iter_counter)
                    max_ROCAUC_model = model.state_dict()
                    embeds = model.embed()
                    max_ROCAUC_model_embed = embeds
                    max_ROCAUC_model_on_test = ROCAUC_test

                log.info(
                    ('[%s] Eval: {'
                     '"iter": %d, '
                     '"loss": %.6f, '
                     '"elapsed (for %d iter.)": %.2f, '
                     '"elapsed (for eval.)": %.2f, '
                     '"rocauc_train": %.6f, '
                     '"rocauc_valid": %.6f, '
                     '"rocauc_test": %.6f, '
                     '"best_rocauc_valid": %.6f, '
                     '"best_rocauc_valid_iter": %d, '
                     '"best_rocauc_valid_test": %.6f, '
                     '}') % (
                         opt["exp_name"], iter_counter, np.mean(train_loss), opt["eval_each"], timeit.default_timer() - t_start, eval_elapsed,
                         ROCAUC_train, ROCAUC_valid, ROCAUC_test, max_ROCAUC[0],  max_ROCAUC[1], max_ROCAUC_model_on_test,
                    )
                )

                former_loss = np.mean(train_loss)
                train_loss = []
                t_start = timeit.default_timer()
                if iter_counter < opt["iter"]:
                    pbar = tqdm(total=opt["eval_each"])

            if iter_counter >= opt["iter"]:
                log.info(
                    ('[%s] RESULT: {'
                    '"best_rocauc_valid": %.6f, '
                    '"best_rocauc_valid_test": %.6f, '
                    '}') % (
                         opt["exp_name"],
                         max_ROCAUC[0], max_ROCAUC_model_on_test,
                         )
                )

                print(""" save model """)
                embeds = model.embed()

                torch.save({
                    'model': model.state_dict(),
                    'node2id': dataset.node2id,
                    'data_vectors': dataset.data_vectors,
                    'embeds_at_final_iteration': embeds,
                    'best_rocauc_model' : max_ROCAUC_model,
                    'best_rocauc_valid' : max_ROCAUC[0],
                    'best_rocauc_valid_embeds' : max_ROCAUC_model_embed,
                    'best_rocauc_valid_test' : max_ROCAUC_model_on_test,
                    'best_rocauc_valid_iteration': max_ROCAUC[1],
                    'total_iteration': iter_counter,
                }, f'{opt["save_dir"]}/{opt["exp_name"]}.pth')
                sys.exit()


def evaluation(model, neighbor_train, neighbor_valid, neighbor_test, task, log, neproc, cuda=False, verbose=False):
    t_start = timeit.default_timer()

    ips_weight = None

    embeds = model.embed()
    if model.model == "WIPS":
        ips_weight = model.get_ips_weight()
        log.info("WIPS's ips weight's ratio : pos {}, neg {}".format(np.sum(ips_weight>=0),np.sum(ips_weight<0)))

    neighbor_train = list(neighbor_train.items())
    chunk = int(len(neighbor_train)/neproc + 1)
    queue = mp.Manager().Queue()
    processes = []
    for i in range(neproc):
            p = mp.Process(
                target=eval_thread,
                args=(neighbor_train[i*chunk:(i+1)*chunk], model, embeds, ips_weight, queue, cuda, i==0 and verbose)
            )
            p.start()
            processes.append(p)
    rocauc_scores_train = list()
    for i in range(neproc):
        rocauc_score = queue.get()
        rocauc_scores_train += rocauc_score

    rocauc_scores_valid = rocauc_scores_train.copy()
    rocauc_scores_test = rocauc_scores_train.copy()

    if neighbor_valid is not None:
        neighbor_valid = list(neighbor_valid.items())
        chunk = int(len(neighbor_valid)/neproc + 1)
        queue = mp.Manager().Queue()
        processes = []
        for i in range(neproc):
            p = mp.Process(
                target=eval_thread,
                args=(neighbor_valid[i*chunk:(i+1)*chunk], model, embeds, ips_weight, queue, cuda, i==0 and verbose)
            )
            p.start()
            processes.append(p)
        rocauc_scores_valid = list()
        for i in range(neproc):
            rocauc_score = queue.get()
            rocauc_scores_valid += rocauc_score

    if neighbor_test is not None:
        neighbor_test = list(neighbor_test.items())
        chunk = int(len(neighbor_test)/neproc + 1)
        queue = mp.Manager().Queue()
        processes = []
        for i in range(neproc):
            p = mp.Process(
                target=eval_thread,
                args=(neighbor_test[i*chunk:(i+1)*chunk], model, embeds, ips_weight, queue, cuda, i==0 and verbose)
            )
            p.start()
            processes.append(p)
        rocauc_scores_test = list()
        for i in range(neproc):
            rocauc_score = queue.get()
            rocauc_scores_test += rocauc_score

    return np.mean(rocauc_scores_train), np.mean(rocauc_scores_valid), np.mean(rocauc_scores_test), timeit.default_timer()-t_start


def eval_thread(neighbor_thread, model, embeds, ips_weight, queue, cuda, verbose):
    embeds = [torch.from_numpy(i) for i in embeds]
    embeddings = []
    with torch.no_grad():
        for i in range(len(embeds)):
            embeddings.append(Variable(embeds[i]))
        if ips_weight is not None:
            ips_weight = Variable(torch.from_numpy(ips_weight))
    rocauc_scores = []
    if verbose : bar = tqdm(desc='Eval', total=len(neighbor_thread), mininterval=1, bar_format='{desc}: {percentage:3.0f}% ({remaining} left)')
    for _s, s_neighbor in neighbor_thread:
        if verbose : bar.update()
        s = torch.tensor(_s)
        target_embeddings = []
        with torch.no_grad():
            for i in range(len(embeds)):
                target_embeddings.append(Variable(embeds[i][s].expand_as(embeddings[i])))
        if cuda:
            input_embeddings = target_embeddings + embeddings
            if ips_weight is not None:
                _dists = model.distfn(input_embeddings, w=ips_weight).data.cpu().numpy().flatten()
            else:
                _dists = model.distfn(input_embeddings).data.cpu().numpy().flatten()
            node_num = model.total_node_num
        else:
            input_embeddings = target_embeddings + embeddings
            if ips_weight is not None:
                _dists = model.distfn(input_embeddings, w=ips_weight).data.numpy().flatten()
            else:
                _dists = model.distfn(input_embeddings).data.numpy().flatten()
            node_num = model.total_node_num
        _dists[s] = 1e+12
        _labels = np.zeros(node_num)
        for o in s_neighbor:
            o = torch.tensor(o)
            _labels[o] = 1
        _rocauc_scores = roc_auc_score(_labels, -_dists)
        rocauc_scores.append(_rocauc_scores)
    if verbose : bar.close()
    queue.put(rocauc_scores)
