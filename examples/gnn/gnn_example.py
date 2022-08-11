import torch
from wholegraph.torch import wholegraph_pytorch as wm
from wm_torch import graph_ops as graph_ops
from wm_torch import comm as comm
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
from optparse import OptionParser
import datetime
import time

import apex
from apex.parallel import DistributedDataParallel as DDP

from mpi4py import MPI

parser = OptionParser()
parser.add_option('-g', '--graph_saved_dir', dest='graph_saved_dir', default='wm_data',
                  help="converted graph directory.")
parser.add_option('--name', dest='graph_name', default='papers100m', help='graph name')
parser.add_option('-e', '--epochs', type='int', dest="epochs", default=24,
                  help='number of epochs')
parser.add_option('-b', '--batchsize', type='int', dest="batchsize", default=1024,
                  help='batch size')
parser.add_option('-c', '--classnum', type='int', dest="classnum", default=172,
                  help='class number')
parser.add_option('-n', '--neighbors', dest="neighbors", default='30,30,30',
                  help='train neighboor sample count')
parser.add_option('--hiddensize', type='int', dest="hiddensize", default=256,
                  help='hidden size')
parser.add_option('-l', '--layernum', type='int', dest="layernum", default=3,
                  help='layer number')
parser.add_option('-m', '--model', dest="model", default='sage',
                  help='model type, valid values are: sage, gcn, gat')
parser.add_option('-f', '--framework', dest="framework", default='wg',
                  help='framework type, valid values are: dgl, pyg, wg')
parser.add_option('--heads', type='int', dest="heads", default=1,
                  help='num heads')
parser.add_option('-s', '--inferencesample', type='int', dest="inferencesample", default=30,
                  help='inference sample count, -1 is all')
parser.add_option('-w', '--dataloaderworkers', type='int', dest="dataloaderworkers", default=8,
                  help='number of workers for dataloader')
parser.add_option('-d', '--dropout', type='float', dest="dropout", default=0.5, help='dropout')
parser.add_option('--lr', type='float', dest="lr", default=0.003,
                  help='learning rate')
'''
parser.add_option("--local_rank", type=int, dest="local_rank", default=os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', 0))
parser.add_option("--world_size", type=int, dest="world_size", default=int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1)
'''
(options, args) = parser.parse_args()


def parse_max_neighbors(num_layer, neighbor_str):
    neighbor_str_vec = neighbor_str.split(',')
    max_neighbors = []
    for ns in neighbor_str_vec:
        max_neighbors.append(int(ns))
    assert len(max_neighbors) == 1 or len(max_neighbors) == num_layer
    if len(max_neighbors) != num_layer:
        for i in range(1, num_layer):
            max_neighbors.append(max_neighbors[0])
    #max_neighbors.reverse()
    return max_neighbors


if options.framework == 'dgl':
    import dgl
    from dgl.nn.pytorch.conv import SAGEConv, GATConv
elif options.framework == 'pyg':
    from torch_sparse import SparseTensor
    from torch_geometric.nn import SAGEConv, GATConv
elif options.framework == 'wg':
    from wm_torch.gnn.SAGEConv import SAGEConv
    from wm_torch.gnn.GATConv import GATConv


def get_train_step(sample_count, epochs, batch_size, global_size):
    return sample_count * epochs // (batch_size * global_size)


def create_train_dataset(data_tensor_dict, rank, size):
    return DataLoader(dataset=graph_ops.NodeClassificationDataset(data_tensor_dict, rank, size),
                      batch_size=options.batchsize, shuffle=True, num_workers=options.dataloaderworkers, pin_memory=True)


def create_valid_test_dataset(data_tensor_dict):
    return DataLoader(dataset=graph_ops.NodeClassificationDataset(data_tensor_dict, 0, 1),
                      batch_size=(options.batchsize + 3) // 4, shuffle=False, pin_memory=True)


def create_gnn_layers(in_feat_dim, hidden_feat_dim, class_count, num_layer, num_head):
    gnn_layers = torch.nn.ModuleList()
    for i in range(num_layer):
        layer_output_dim = hidden_feat_dim // num_head if i != num_layer - 1 else class_count
        layer_input_dim = in_feat_dim if i == 0 else hidden_feat_dim
        mean_output = True if i == num_layer - 1 else False
        if options.framework == 'pyg':
            if options.model == 'sage':
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim))
            elif options.model == 'gat':
                concat = not mean_output
                gnn_layers.append(GATConv(layer_input_dim, layer_output_dim, heads=num_head, concat=concat))
            else:
                assert options.model == 'gcn'
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim, root_weight=False))
        elif options.framework == 'dgl':
            if options.model == 'sage':
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim, 'mean'))
            elif options.model == 'gat':
                gnn_layers.append(GATConv(layer_input_dim, layer_output_dim, num_heads=num_head, allow_zero_in_degree=True))
            else:
                assert options.model == 'gcn'
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim, 'gcn'))
        elif options.framework == 'wg':
            if options.model == 'sage':
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim))
            elif options.model == 'gat':
                gnn_layers.append(GATConv(layer_input_dim, layer_output_dim, num_heads=num_head, mean_output=mean_output))
            else:
                assert options.model == 'gcn'
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim, aggregator='gcn'))
    return gnn_layers


def create_sub_graph(target_gid, target_gid_1, edge_data, csr_row_ptr, csr_col_ind, sample_dup_count, add_self_loop: bool):
    if options.framework == 'pyg':
        neighboor_dst_unique_ids = csr_col_ind
        neighboor_src_unique_ids = edge_data[1]
        target_neighbor_count = target_gid.size()[0]
        if add_self_loop:
            self_loop_ids = torch.arange(0, target_gid_1.size()[0], dtype=neighboor_dst_unique_ids.dtype, device=target_gid.device)
            edge_index = SparseTensor(row=torch.cat([neighboor_src_unique_ids, self_loop_ids]).long(),
                                      col=torch.cat([neighboor_dst_unique_ids, self_loop_ids]).long(),
                                      sparse_sizes=(target_gid_1.size()[0], target_neighbor_count))
        else:
            edge_index = SparseTensor(row=neighboor_src_unique_ids.long(), col=neighboor_dst_unique_ids.long(),
                                      sparse_sizes=(target_gid_1.size()[0], target_neighbor_count))
        return edge_index
    elif options.framework == 'dgl':
        if add_self_loop:
            self_loop_ids = torch.arange(0, target_gid_1.numel(), dtype=edge_data[0].dtype, device=target_gid.device)
            block = dgl.create_block((torch.cat([edge_data[0], self_loop_ids]),
                                      torch.cat([edge_data[1], self_loop_ids])), num_src_nodes=target_gid.size(0), num_dst_nodes=target_gid_1.size(0))
        else:
            block = dgl.create_block((edge_data[0], edge_data[1]), num_src_nodes=target_gid.size(0), num_dst_nodes=target_gid_1.size(0))
        return block
    else:
        assert options.framework == 'wg'
        return [csr_row_ptr, csr_col_ind, sample_dup_count]
    return None


def layer_forward(layer, x_feat, x_target_feat, sub_graph):
    if options.framework == 'pyg':
        x_feat = layer((x_feat, x_target_feat), sub_graph)
    elif options.framework == 'dgl':
        x_feat = layer(sub_graph, (x_feat, x_target_feat))
    elif options.framework == 'wg':
        x_feat = layer(sub_graph[0], sub_graph[1], sub_graph[2], x_feat, x_target_feat)
    return x_feat


class HomoGNNModel(torch.nn.Module):
    def __init__(self, graph: graph_ops.HomoGraph, num_layer, hidden_feat_dim, class_count, max_neighbors: str):
        super().__init__()
        self.graph = graph
        self.num_layer = num_layer
        self.hidden_feat_dim = hidden_feat_dim
        self.max_neighbors = parse_max_neighbors(num_layer, max_neighbors)
        self.class_count = class_count
        num_head = options.heads if (options.model == 'gat') else 1
        assert hidden_feat_dim % num_head == 0
        in_feat_dim = self.graph.node_feat_shape()[1]
        self.gnn_layers = create_gnn_layers(in_feat_dim, hidden_feat_dim, class_count, num_layer, num_head)
        self.mean_output = True if options.model == 'gat' else False
        self.add_self_loop = True if options.model == 'gat' else False

    def forward(self, ids):
        ids = ids.to(self.graph.id_type()).cuda()
        target_gids, edge_indice, csr_row_ptrs, csr_col_inds, sample_dup_counts = self.graph.unweighted_sample_without_replacement(
            ids, self.max_neighbors)
        x_feat = self.graph.gather(target_gids[0])
        for i in range(self.num_layer):
            x_target_feat = x_feat[:target_gids[i+1].numel()]
            sub_graph = create_sub_graph(target_gids[i], target_gids[i+1], edge_indice[i], csr_row_ptrs[i], csr_col_inds[i], sample_dup_counts[i], self.add_self_loop)
            x_feat = layer_forward(self.gnn_layers[i], x_feat, x_target_feat, sub_graph)
            if i != self.num_layer - 1:
                if options.framework == 'dgl':
                    x_feat = x_feat.flatten(1)
                x_feat = F.relu(x_feat)
                x_feat = F.dropout(x_feat, options.dropout, training=self.training)
        if options.framework == 'dgl' and self.mean_output:
            out_feat = x_feat.mean(1)
        else:
            out_feat = x_feat
        return out_feat


def valid_test(dataloader, model, name):
    total_correct = 0
    total_valid_sample = 0
    print('%s...' % (name, ))
    for i, (idx, label) in enumerate(dataloader):
        label = torch.reshape(label, (-1,)).cuda()
        model.eval()
        logits = model(idx)
        pred = torch.argmax(logits, 1)
        correct = (pred == label).sum()
        total_correct += correct.cpu()
        total_valid_sample += label.shape[0]
    print('[%s] [%s] accuracy=%5.2f%%' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), name, 100.0 * total_correct / total_valid_sample))


def valid(valid_dataloader, model):
    valid_test(valid_dataloader, model, 'VALID')


def test(test_data, model):
    test_dataloader = create_valid_test_dataset(data_tensor_dict=test_data)
    valid_test(test_dataloader, model, 'TEST')


def train(train_data, valid_data, model, optimizer):
    print('start training...')
    train_dataloader = create_train_dataset(data_tensor_dict=train_data, rank=comm.get_rank(), size=comm.get_world_size())
    valid_dataloader = create_valid_test_dataset(data_tensor_dict=valid_data)
    total_steps = get_train_step(len(train_data['idx']), options.epochs, options.batchsize, comm.get_world_size())
    print('total_steps=%d' % (total_steps, ))
    train_step = 0
    epoch = 0
    loss_fcn = torch.nn.CrossEntropyLoss()
    skip_8_epoch_time = 0
    train_start_time = time.time()
    while train_step < total_steps:
        if epoch == 1:
            skip_8_epoch_time = time.time()
        for i, (idx, label) in enumerate(train_dataloader):
            if train_step >= total_steps:
                break
            label = torch.reshape(label, (-1,)).cuda()
            optimizer.zero_grad()
            model.train()
            logits = model(idx)
            loss = loss_fcn(logits, label)
            loss.backward()
            optimizer.step()
            if comm.get_rank == 0 and train_step % 100 == 0:
                print('[%s] [LOSS] step=%d, loss=%f' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), train_step, loss.cpu().item()))
            train_step = train_step + 1
        epoch = epoch + 1
    comm.synchronize()
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    if comm.get_rank() == 0:
        print('[%s] [TRAIN_TIME] train time is %.2f seconds' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), train_time))
        print('[EPOCH_TIME] %.2f seconds' % ((train_end_time - skip_8_epoch_time) / (options.epochs - 8), ))
    if comm.get_rank() == 0:
        valid(valid_dataloader, model)


def main():
    wm.init_lib()
    torch.set_num_threads(1)
    comma = MPI.COMM_WORLD
    shared_comma = comma.Split_type(MPI.COMM_TYPE_SHARED)
    os.environ["RANK"] = str(comma.Get_rank())
    os.environ["WORLD_SIZE"] = str(comma.Get_size())
    # slurm in Selene has MASTER_ADDR env
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = 'localhost'
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = '12335'
    local_rank = shared_comma.Get_rank()
    print("Rank=%d, local_rank=%d" % (local_rank, comma.Get_rank()))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    wm.mp_init(shared_comma.Get_rank(), shared_comma.Get_size())
    if comma.Get_rank() == 0:
        print('Framework=%s, Model=%s' % (options.framework, options.model))

    train_data, valid_data, test_data = graph_ops.load_pickle_data(options.graph_saved_dir, options.graph_name)

    dist_homo_graph = graph_ops.HomoGraph()
    use_chunked = True
    dist_homo_graph.load(options.graph_saved_dir, options.graph_name, use_chunked, False)
    print('Rank=%d, Graph loaded.' % (comma.Get_rank(), ))
    model = HomoGNNModel(dist_homo_graph, options.layernum, options.hiddensize, options.classnum, options.neighbors)
    print('Rank=%d, model created.' % (comma.Get_rank(), ))
    model.cuda()
    print('Rank=%d, model movded to cuda.' % (comma.Get_rank(), ))
    model = DDP(model, delay_allreduce=True)
    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=options.lr)
    print('Rank=%d, optimizer created.' % (comma.Get_rank(), ))

    train(train_data, valid_data, model, optimizer)
    if comm.get_rank() == 0:
        test(test_data, model)

    wm.finalize_lib()
    print('Rank=%d, wholegraph shutdown.' % (comma.Get_rank(), ))


if __name__ == "__main__":
    main()

