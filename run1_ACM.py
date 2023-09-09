import time
import argparse

import torch.nn.functional as F
import torch.sparse
import numpy as np
import dgl

import utils_ACM
from model_ACM import StarGAT_nc

# hyper parameters
lr = 0.005
weight_decay = 0.001
dropout_rate = 0.5
out_dim = 3
num_ntype = 3
dataset_str = 'ACM'

def run_model(hidden_dim, num_heads, attn_vec_dim, num_epochs, patience, repeat, save_postfix, batch_size, neighbor_samples):
    
    labels, train_idx, val_idx, test_idx, type_mask, nx_G_list, edge_list, adjM = utils_ACM.load_data(dataset_str)    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    features_list = []
    in_dims = []
    for i in range(num_ntype):
        dim = (type_mask == i).sum()
        in_dims.append(dim)
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(dim))
        features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)) 

    labels = torch.LongTensor(labels).to(device)
    adjM = torch.FloatTensor(adjM).to(device)
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)
        
    for _ in range(repeat):
        net = StarGAT_nc(in_dims, hidden_dim, num_heads, out_dim, attn_vec_dim, features_list[0].shape[0], dropout_rate)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        net.train()
        early_stopping = utils_ACM.EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        train_idx_generator = utils_ACM.index_generator(batch_size=batch_size, indices=train_idx)
        val_idx_generator = utils_ACM.index_generator(batch_size=batch_size, indices=val_idx, shuffle=False)
        
        for epoch in range(num_epochs):
            t_start = time.time()

            # training forward
            net.train()
            for iteration in range(train_idx_generator.num_iterations()):
                t0 = time.time()
                train_idx_batch = train_idx_generator.next()
                train_idx_batch.sort()
                train_g_list, train_indices_list, train_idx_batch_mapped_list = utils_ACM.parse_minibatch(
                    nx_G_list, edge_list, train_idx_batch, device, neighbor_samples)
                
                train_g_list_temp = []
                for g in train_g_list:
                    g_temp = g.to(device)
                    train_g_list_temp.append(g_temp)  

                temp_train.extend(train_idx_batch)
                t1 = time.time()
                dur1.append(t1 - t0)
                logits, embeddings = net((train_g_list_temp, features_list, type_mask, train_indices_list, train_idx_batch_mapped_list, adjM, train_idx_batch)) 
                
                logp = F.log_softmax(logits, 1)
                train_loss = F.nll_loss(logp, labels[train_idx_batch])   

                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur3.append(t3 - t2)                
            
                # print training info
                if iteration % 50 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))

            # validation forward
            net.eval()
            with torch.no_grad():
                val_logp = []
                temp_val = []
                for iteration in range(val_idx_generator.num_iterations()):
                    val_idx_batch = val_idx_generator.next()
                    val_g_list, val_indices_list, val_idx_batch_mapped_list = utils_ACM.parse_minibatch(
                        nx_G_list, edge_list, val_idx_batch, device, neighbor_samples)
                  
                    val_g_list_temp = []
                    for g in val_g_list:
                        g_temp = g.to(device)
                        val_g_list_temp.append(g_temp)                        

                    temp_val.extend(val_idx_batch)
                    logits, embeddings = net((val_g_list_temp, features_list, type_mask, val_indices_list, val_idx_batch_mapped_list, adjM, val_idx_batch)) 
                    logp = F.log_softmax(logits, 1)
                    val_logp.append(logp)
                val_loss = F.nll_loss(torch.cat(val_logp, 0), labels[temp_val])    
       
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break


        # evaluation
        test_idx_generator = utils_ACM.index_generator(batch_size=batch_size, indices=test_idx, shuffle=False)
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        print('evaluation results')
        net.eval()
        test_embeddings = []
        test_logits = []
        temp_test = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_g_list, test_indices_list, test_idx_batch_mapped_list = utils_ACM.parse_minibatch(nx_G_list, edge_list, test_idx_batch,device, neighbor_samples)
 
                test_g_list_temp = []
                for g in test_g_list:
                    g_temp = g.to(device)
                    test_g_list_temp.append(g_temp)                  

                temp_test.extend(test_idx_batch)
                logits, embeddings = net((test_g_list_temp, features_list, type_mask, test_indices_list, test_idx_batch_mapped_list, adjM, test_idx_batch))
                test_embeddings.append(embeddings)
                test_logits.append(logits)
                
            test_embeddings = torch.cat(test_embeddings, 0)
            test_logits = torch.cat(test_logits, 0)
                    
            temp_labels = []
            for i in temp_test:
                temp_labels.append(labels[i])
            temp_labels = torch.tensor(temp_labels, device=device)            
            
            utils_ACM.evaluate_results_nc(test_embeddings.cpu().detach().numpy(), temp_labels.cpu().detach().numpy(), num_ntype, test_logits.cpu().detach())
            

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='StarGAT testing for ACM dataset')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--epoch', type=int, default=10, help='Number of epochs. Default is 20.')
    ap.add_argument('--patience', type=int, default=10, help='Patience. Default is 5.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='ACM', help='Postfix for the saved model and result. Default is AC<.')
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=None, help='Number of neighbors sampled. Default is 100.')

    args = ap.parse_args()
    run_model(args.hidden_dim, args.num_heads, args.attn_vec_dim, args.epoch, args.patience, args.repeat, args.save_postfix, args.batch_size, args.samples)