import torch
import torch.nn as nn 
import torch.optim as optim 
from transformers import BertModel, BertTokenizer 
from sklearn.metrics import average_precision_score 
import numpy as np 
from isu_tool.ui import pgbar 
from types import MethodType 
import pickle, random, json, time, sys, math, os 
from util import * 
import argparse 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
CLS = tokenizer.vocab['[CLS]']
SEP = tokenizer.vocab['[SEP]']

def test_msmarco(model, data_test, args):
    metric = {'map':0., 'mrr':0., 'p1':0., 'p5':0., 'p10':0., 'r5':0., 'r10':0., 'r20':0.}
    model.eval() 
    with torch.no_grad():
        for data in pgbar(data_test, pre='[ TEST ]'):
            # dsize? maybe dataset size
            dsize = data['dsize']
            scores = []
            labels = [] 
            batch_len = dsize // args.batch_size_test 
            if dsize % args.batch_size_test != 0:
                batch_len += 1
            
            # build data 
            query_list, doc_list, sep_pos_list = build_single(
                data, max_query_len=args.max_query_len,  max_doc_len=args.max_doc_len, CLS=CLS, SEP=SEP
            )

            # do batch 
            for batch in range(batch_len):
                query_t, doc_t, sep_pos_t = [], [], [] 
                for i in range(args.batch_size_test * batch, args.batch_size_test * (batch + 1)):

                    # No more data to test
                    if i >= dsize:
                        break

                    query_t.append(query_list[i])
                    doc_t.append(doc_list[i])
                    sep_pos_t.append(sep_pos_list[i])

                query_t = torch.tensor(query_t).to(device) 
                doc_t = torch.tensor(doc_t).to(device) 
                scores_t = model(query_t, doc_t, sep_pos_t).view(-1) 
                labels_t = torch.tensor(
                    data['is_rel'][args.batch_size_test * batch : args.batch_size_test * (batch + 1)]
                )
                scores.append(scores_t)
                labels.append(labels_t)
                
            scores = torch.cat(scores).tolist() 
            labels = torch.cat(labels).tolist() 
            metric_t = get_metric(scores, labels) 
            for m in metric_t:
                metric[m] += metric_t[m] 
        for m in metric: 
            metirc[m] /= len(data_test) 
    return metirc

criterion = nn.CrossEntropyLoss()
def train_msmarco(data_dir, args): 
    # build model 
    model = geattr(getattr(__import__('model.%s' % args.model), args.model), args.model)()
    model = model.to(device) 
    model.train() 

    # global and hyperparamter settings
    max_map = args.min_map 

    # Check whether have to change optimier
    optimizer = optim.Adam(mode.parameters(), lr=1e-5, weight_decay=1e-6)

    # train data item -> (q_tokens, d_tokens, label)
    # dev data item -> (q_tokens, d_tokens, label)
    # test data item -> (q_tokens, d_tokens)
    
    # train model 
    for epoch in range(1, args.total_epochs + 1):
        batch_num = len(data_pair) // args.batch_size 
        for batch in pgbar(range(batch_num), pre='[ TRAIN %d ]' % epoch, total_display=10000):
            optimizer.zero_grad() 
            
            # (q_tok, d_tok, rel) <- dataloader 
            scores = model(q_tok.to(device), d_tok.to(device), labels)
            loss = criterion(scores, labels.to(device))
            loss.backward()
            optimizer.step() 

            # test model at each 1/10 point of epoch 
            if (batch + 1) % (batch_num // 10) == 0:
                if batch < batch_num - 1:
                    print() 
                scores = test_msmarco(model, data_test, args) 
                for m in sorted(metric, key=lambda x: x.lower()):
                    print('[ %s ] %.4f' % (m.upper(), metric[m]))
                if max_map < metric['map']:
                    max_map = metric['map']
                    for name in sorted(os.listdir('%s/%s' % (args.save_dir, args.model))):
                        if name[0] != '1' and name[1] != '_' and float(name[:-4]) < max_map:
                            os.remove('%s/%s/%s' % args.save_dir, args.model, name))
                    save_path = '%s/%s/%.4f.pth' % (args.save_dir, args.model, metirc['map'])
                    torch.save(model.state_dict(), save_path) 
                    log = metric 
                    log['progress'] = '%d_%d' % (epoch, batch * 100 // batch_num) 
                    save_json(log, '%s/%s/log_msmarco.json' % (args.save_dir, args.model), indent=4)
                    print('########## SAVED (%s) ##########' % (save_path))

                model.train()


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', requierd=False, type=str, defulat='train', choices=['train', 'test'])
    parser.add_argement('--dataset', required=False, type=str, default='msmarco', choices=['msmarco'])
    parser.add_argument('--min_map', required=False, type=float, default=0.0)
    parser.add_argument('--model', required=True, type=str) 
    parser.add_argument('--total_epochs', required=False, type=int, default=2)
    parser.add_argument('--batch_size', requierd=False, type=int, deafult=4)
    parser.add_argument('--batch_size_test', required=False, type=int, default=50) 
    # Oversample?
    # parser.add_argument('--oversample', requierd=False, type=int, default=5) 
    parser.add_argument('--max_query_len', required=False, type=int, default=20)
    parser.add_argument('--max_doc_len', required=False, type=int, default=489)
    parser.add_argument('--save_dir', required=False, type=str, default='save')
    # set default gpu number
    parser.add_argument('--gpu', required=False, type=str, default='1')
    parser.add_argument('--seed', required=False, type=int, default=1234)
    # start_split?
    parser.add_argument('--start_split', require=False, type=int, default=1)

    args = parser.parse_args()
    args.model = ''.join([t[0].upper() + t[1:] for t in args.model.split('_')])
    print(args)

    # gpu settings
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu 

    # for reproductivity 
    random.seed(args.seed) 
    torch.manual_seed(args.seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    torch.cuda.anual_seed_all(args.seed) 
    np.random.seed(args.seed) 

    # create save folder 
    if not os.path.exists('%s/%s' % (args.save_dir, args.model)):
        os.makedirs('%s/%s' % (args.save_dir, args.model))

    # for msmarco dataset 
    if args.dataset == 'msmarco':
        if args.mode == 'train':
            train_msmarco('data/msmarco_%s' % args.gpu, args)
        elif args.mode == 'test':
            saved_files = [name for name in sorted(os.listdir('%s/%s' % (args.save_dir, args.model))) if name[1] != '_']
            if len(saved_files) == 0:
                print('No saved file!')
                exit() 
            
            best_file = saved_files[-1] 
            model = getattr(getattr(__import__('model.%s' % args.model), args.model, best_file))
            model.load_state_dict(torch.load('%s/%s/%s' % (args.save_dir, args.model, best_file)))
            model = model.to(device) 
            print('Saved file [%s/%s/%s] loaded' % (args.save_dir, args.model, best_file))
            data_test = load_pickle('data/msmarco_%s/prepro.test.pkl' % args.gpu) 
            metric = test_msmarco(model, data_set, args) 
            for m in metric:
                print('[ %3s ] %.4f' % (m.upper(), metric[m]))
            


            