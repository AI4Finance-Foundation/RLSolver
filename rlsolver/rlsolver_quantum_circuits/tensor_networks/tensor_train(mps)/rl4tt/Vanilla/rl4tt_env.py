import sys
import torch as th
from math import log10 as math_log10

TEN = th.Tensor



def get_nodes_list_of_tensor_train(len_list: int = 4):
    nodes = [[] for _ in range(len_list)]  # 初始化邻接表
    for i in range(len_list):
        if i > 0:
            nodes[i].append(i - 1)
        if i < len_list - 1:
            nodes[i].append(i + 1)
        nodes[i].append(i + len_list)
        nodes.append([i])
    return nodes


def get_nodes_ary(nodes_list: list) -> TEN:
    # nodes_list = NodesSycamore
    nodes_ary = th.zeros((len(nodes_list), max([len(nodes) for nodes in nodes_list])), dtype=th.int) - 1
    # # -1 表示这里没有连接
    for i, nodes in enumerate(nodes_list):
        for j, node in enumerate(nodes):
            nodes_ary[i, j] = node
    return nodes_ary


def get_edges_ary(nodes_ary: TEN) -> TEN:
    edges_ary = th.zeros_like(nodes_ary, dtype=nodes_ary.dtype)
    edges_ary[nodes_ary >= 0] = -2  # -2 表示这里的 edge_i 需要被重新赋值
    edges_ary[nodes_ary == -1] = -1  # -1 表示这里的 node 没有连接另一个 node

    num_edges = 0
    '''get nodes_ary'''
    # for i, nodes in enumerate(nodes_ary):  # i 表示节点的编号
    #     for j, node in enumerate(nodes):  # node 表示跟编号为i的节点相连的另一个节点
    #         edge_i = edges_ary[i, j]
    #         if edge_i == -2:
    #             _j = th.where(nodes_ary[node] == i)
    #             edges_ary[i, j] = num_edges
    #             edges_ary[node, _j] = num_edges
    #             num_edges += 1
    '''get nodes_ary and sort the ban edges to large indices'''
    for i, nodes in list(enumerate(nodes_ary))[::-1]:  # i 表示节点的编号
        for j, node in enumerate(nodes):  # node 表示跟编号为i的节点相连的另一个节点
            edge_i = edges_ary[i, j]
            if edge_i == -2:
                nodes_ary_node: TEN = nodes_ary[node]
                _j = th.where(nodes_ary_node == i)

                edges_ary[i, j] = num_edges
                edges_ary[node, _j] = num_edges
                num_edges += 1
    _edges_ary = edges_ary.max() - edges_ary
    _edges_ary[edges_ary == -1] = -1
    edges_ary = _edges_ary
    return edges_ary


def get_node_dims_arys(nodes_ary: TEN) -> list:
    num_nodes = nodes_ary.shape[0]

    arys = []
    for nodes in nodes_ary:
        positive_nodes = nodes[nodes >= 0].long()
        ary = th.zeros((num_nodes,), dtype=th.int)  # 因为都是2，所以我用0 表示 2**0==1
        ary[positive_nodes] = 1  # 2量子比特门，这里的计算会带来2个单位的乘法，因为都是2，所以我用1 表示 2**1==2
        arys.append(ary)
    return arys


def get_node_bool_arys(nodes_ary: TEN) -> list:
    num_nodes = nodes_ary.shape[0]

    arys = []
    for i, nodes in enumerate(nodes_ary):
        ary = th.zeros((num_nodes,), dtype=th.bool)
        ary[i] = True
        arys.append(ary)
    return arys


class TensorNetworkEnv:
    def __init__(self, nodes_list: list, device: th.device):
        self.device = device

        '''build node_arys and edges_ary'''
        nodes_ary = get_nodes_ary(nodes_list)
        num_nodes = nodes_ary.max().item() + 1
        assert num_nodes == nodes_ary.shape[0]

        edges_ary = get_edges_ary(nodes_ary)
        num_edges = edges_ary.max().item() + 1
        assert num_edges == (edges_ary != -1).sum() / 2

        # self.nodes_ary = nodes_ary
        self.edges_ary = edges_ary.to(device)
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.ban_edges = None  # todo not elegant

        '''build for get_log10_multiple_times'''
        node_dims_arys = get_node_dims_arys(nodes_ary)
        assert num_edges == sum([(ary == 1).sum().item() for ary in node_dims_arys]) / 2

        node_bool_arys = get_node_bool_arys(nodes_ary)
        assert num_nodes == sum([ary.sum() for ary in node_bool_arys])

        self.node_dims_ten = th.stack(node_dims_arys).to(device)
        self.node_bool_ten = th.stack(node_bool_arys).to(device)

    def get_log10_multiple_times(self, edge_argsorts: TEN) -> TEN:
        # edge_argsort = th.rand(self.num_edges).argsort()
        device = self.device
        edges_ary: TEN = self.edges_ary
        num_envs, run_edges = edge_argsorts.shape
        assert run_edges == self.num_edges - self.ban_edges
        vec_env_is = th.arange(num_envs, device=device)

        node_dims_tens = th.stack([self.node_dims_ten.clone() for _ in range(num_envs)])
        node_bool_tens = th.stack([self.node_bool_ten.clone() for _ in range(num_envs)])

        mult_pow_timess = th.zeros((num_envs, run_edges), dtype=th.float64, device=device)

        for i in range(run_edges):
            edge_is = edge_argsorts[:, i]

            """Vanilla (single)"""
            for j in range(num_envs):
                edge_i = edge_is[j]
                node_dims_arys = node_dims_tens[j]
                node_bool_arys = node_bool_tens[j]

                '''find two nodes of an edge_i'''
                node_i0, node_i1 = th.where(edges_ary == edge_i)[0]  # 找出这条edge 两端的node
                # assert isinstance(node_i0.item(), int)
                # assert isinstance(node_i1.item(), int)

                '''calculate the multiple and avoid repeat'''
                contract_dims = node_dims_arys[node_i0] + node_dims_arys[node_i1]  # 计算收缩后的node 的邻接张量的维度 以及来源
                contract_bool = node_bool_arys[node_i0] | node_bool_arys[node_i1]  # 计算收缩后的node 由哪些原初node 合成
                # assert contract_dims.shape == (num_nodes, )
                # assert contract_bool.shape == (num_nodes, )

                # 收缩掉的edge 只需要算一遍乘法。因此上面对 两次重复的指数求和后乘以0.5
                mult_pow_time = contract_dims.sum(dim=0) - (contract_dims * contract_bool).sum(dim=0) * 0.5
                # assert mult_pow_time.shape == (1, )
                mult_pow_timess[j, i] = mult_pow_time

                '''adjust two list: node_dims_arys, node_bool_arys'''
                contract_dims[contract_bool] = 0  # 把收缩掉的边的乘法数量赋值为2**0，接下来不再参与乘法次数的计算

                node_dims_arys[contract_bool] = contract_dims.repeat(1, 1)  # 根据 bool 将所有收缩后的节点都刷新成相同的信息
                node_bool_arys[contract_bool] = contract_bool.repeat(1, 1)  # 根据 bool 将所有收缩后的节点都刷新成相同的信息

                # print('\n;;;', i, edge_i, node_i0, node_i1)
                # [print(ary) for ary in node_dims_arys[:-self.ban_edges]]
                # [print(ary.int()) for ary in node_bool_arys[:-self.ban_edges]]

            # """Vectorized"""
            # '''find two nodes of an edge_i'''
            # vec_edges_ary: TEN = edges_ary[None, :, :]
            # vec_edges_is: TEN = edge_is[:, None, None]
            # res = th.where(vec_edges_ary == vec_edges_is)[1]
            # res = res.reshape((num_envs, 2))
            # node_i0s, node_i1s = res[:, 0], res[:, 1]
            # # assert node_i0s.shape == (num_envs, )
            # # assert node_i1s.shape == (num_envs, )
            #
            # '''calculate the multiple and avoid repeat'''
            # contract_dimss = node_dims_tens[vec_env_is, node_i0s] + node_dims_tens[vec_env_is, node_i1s]
            # contract_bools = node_bool_tens[vec_env_is, node_i0s] | node_bool_tens[vec_env_is, node_i1s]
            # # assert contract_dimss.shape == (num_envs, num_nodes)
            # # assert contract_bools.shape == (num_envs, num_nodes)
            #
            # mult_pow_times = contract_dimss.sum(dim=1) - (contract_dimss * contract_bools).sum(dim=1) * 0.5
            # # assert mult_pow_times.shape == (num_envs, )
            # mult_pow_timess[:, i] = mult_pow_times
            #
            # '''adjust two list: node_dims_arys, node_bool_arys'''
            # contract_dimss[contract_bools] = 0  # 把收缩掉的边的乘法数量赋值为2**0，接下来不再参与乘法次数的计算
            #
            # for j in range(num_envs):  # 根据 bool 将所有收缩后的节点都刷新成相同的信息
            #     contract_dims = contract_dimss[j]
            #     contract_bool = contract_bools[j]
            #     node_dims_tens[j, contract_bool] = contract_dims.repeat(1, 1)
            #     node_bool_tens[j, contract_bool] = contract_bool.repeat(1, 1)

            # print('\n;;;', i, )
            # env_i = 0
            # [print(ary) for ary in node_dims_tens[env_i, :-self.ban_edges]]
            # [print(ary.int()) for ary in node_bool_tens[env_i, :-self.ban_edges]]

        # 计算这个乘法个数时，即便用 float64 也偶尔会过拟合，所以先除以 2**temp_power ，求log10 后再恢复它
        max_tmp_power = int(mult_pow_timess.max().item() - 960)  # 1024 is the limit.
        # todo 许伟，我改了两行，被我修改的第一行是上面这一行，用 max - 960 自动设置最大的 max_power 。960 是一个略小于极限 1024的数值
        multiple_times = (2 ** (mult_pow_timess - max_tmp_power)).sum(dim=1)
        multiple_times = multiple_times.log10() + math_log10(2 ** max_tmp_power)  # 用Python原生的math.log10(int) 计算大数
        # todo 许伟，我改了两行，被我修改的第二行是上面这一行，用 math_log10
        return multiple_times.detach()


def run():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    # nodes_list, ban_edges = NodesSycamoreN53M12, 0
    nodes_list, ban_edges = get_nodes_list_of_tensor_train(len_list=200), 200

    env = TensorNetworkEnv(nodes_list=nodes_list, device=device)
    env.ban_edges = ban_edges
    print(f"\nnum_nodes      {env.num_nodes:9}"
          f"\nnum_edges      {env.num_edges:9}"
          f"\nban_edges      {env.ban_edges:9}")
    num_envs = 32

    edge_arys = th.rand((num_envs, env.num_edges - env.ban_edges), device=device)
    # th.save(edge_arys, 'temp.pth')
    # edge_arys = th.load('temp.pth', map_location=device)

    multiple_times = env.get_log10_multiple_times(edge_argsorts=edge_arys.argsort(dim=1))
    print(f"multiple_times(log10) {multiple_times}")


if __name__ == '__main__':
    run()
