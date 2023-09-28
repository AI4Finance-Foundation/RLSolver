import os
import time
import math
import numpy as np
import torch as th

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use('Agg') if os.name != 'nt' else None  # Generating matplotlib graphs without a running X server [duplicate]
except ImportError:
    mpl = None
    plt = None

TEN = th.Tensor


class EncoderBase64:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

        self.base_digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
        self.base_num = len(self.base_digits)

    def bool_to_str(self, x_bool: TEN) -> str:
        x_int = int(''.join([('1' if i else '0') for i in x_bool.tolist()]), 2)

        '''bin_int_to_str'''
        base_num = len(self.base_digits)
        x_str = ""
        while True:
            remainder = x_int % base_num
            x_str = self.base_digits[remainder] + x_str
            x_int //= base_num
            if x_int == 0:
                break

        if len(x_str) > 120:
            x_str = '\n'.join([x_str[i:i + 120] for i in range(0, len(x_str), 120)])
        if len(x_str) > 64:
            x_str = f"\n{x_str}"
        return x_str.zfill(math.ceil(self.num_nodes // 6 + 1))

    def str_to_bool(self, x_str: str) -> TEN:
        x_b64 = x_str.replace('\n', '')

        '''b64_str_to_int'''
        x_int = 0
        base_len = len(x_b64)
        for i in range(base_len):
            digit = self.base_digits.index(x_b64[i])
            power = base_len - 1 - i
            x_int += digit * (self.base_num ** power)

        x_bin: str = bin(x_int)[2:]
        x_bool = th.zeros(self.num_nodes, dtype=th.int8)
        x_bool[-len(x_bin):] = th.tensor([int(i) for i in x_bin], dtype=th.int8)
        return x_bool


class Evaluator:
    def __init__(self, save_dir: str, num_nodes: int, x: TEN, v: int):
        self.start_timer = time.time()
        self.recorder1 = []
        self.recorder2 = []
        self.encoder_base64 = EncoderBase64(num_nodes=num_nodes)

        self.best_x = x  # solution x
        self.best_v = v  # objective value of solution x

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def record1(self, i: float, v: int):
        self.recorder1.append((i, v))

    def record2(self, i: float, v: int, x: TEN):
        self.recorder2.append((i, v))

        if v > self.best_v:
            self.best_x = x
            self.best_v = v
            self.logging_print(v=v, if_show_x=True)

    def plot_record(self, fig_dpi: int = 300):
        if plt is None:
            return

        recorder1 = np.array(self.recorder1) if len(self.recorder1) else np.zeros((1, 2))
        recorder2 = np.array(self.recorder2) if len(self.recorder2) else np.zeros((1, 2))
        np.save(f"{self.save_dir}/recorder1.npy", recorder1)
        np.save(f"{self.save_dir}/recorder2.npy", recorder2)

        plt.plot(recorder1[:, 0], recorder1[:, 1], linestyle='-', label='real time')
        plt.plot(recorder2[:, 0], recorder2[:, 1], linestyle=':', label='back test')
        plt.scatter(recorder2[:, 0], recorder2[:, 1])

        plt.title(f"best_obj_value {self.best_v}")
        plt.legend()
        plt.grid()

        plt.savefig(f"{self.save_dir}/recorder.jpg", dpi=fig_dpi)
        plt.close('all')

    def logging_print(self, v: float, if_show_x: bool = False):
        used_time = int(time.time() - self.start_timer)
        x_str = self.encoder_base64.bool_to_str(self.best_x) if if_show_x else ''
        i = self.recorder2[-1][0]
        print(f"| used_time {used_time:8}  i {i:8}  good_value {v:8}  best_value {self.best_v:8}  {x_str}")


X_G14 = """
11Re2ycMx2zCiEhQl5ey$HyYnkUhDVE6KkPnuuhcWXwUO9Rn1fxrt_cn_g6iZFQex1YpwjD_j7KzbNN71qVekltv3QscNQJjrnrqHfsnOKWJzg9nJhZ$qh69
$X_BvBQirx$i3F
"""  # 3064, SOTA=3064
"""
11Re2ydMx2zCiEhQl5ey$PyYnkUhDVE6KkQnuuhc0XwUO9RnXfxrt_dn_g6aZFQ8x1YpwbD_j7KzaNN71qVuklpv3Q_cNQJjnnrrHjsnOKWIzg9nJxZ$qh69
$n_BHBRirx$i3F
"""  # 3064, SOTA=3064
"""
2_aNz3Of4z2pJnKaGwN30k3TEHXKoWnvhHaE77KPlU5XdsaE_UCA81PE1LvJSmbN4_Ti5Qo1IOh2Aeeu_BWNHGC6yb1GebiIAEAAkI9EdhVj2LsEiKS2BKvs
0E1qkqaJ840Jym
"""  # 3064, SOTA=3064

X_G15 = """
hzvKByHMl4xek23GZucTFBM0f530k4DymcJ5QIcqJyrAoJBkI3g5OaCIpvGsf$l4cLezTm6YOtuDvHtp38hIwUQc3tdTBWocjZj5dX$u1DEA_XX6vESoZz2W
NZpaM3tN$bzhE
"""  # 3050, SOTA=3050
"""
3K26hq3kfGx4N1zylS7HYmqf$Mwy$Hxo3FPiubjPBiBgrDirHj_LwZRpjC6l8I0GxPgN2YBvTb87oMkeCytKj5pbPy8OYyPDPIS2wOS27_onr1UUP6pZDCAV
VeSCVfyCe2Q0Kn
"""  # 3050, SOTA=3050
"""
3K26hq3kfGx4N1zylS7PYmqf$Mwy$nxo3FPiwbjPBi3ArDirHjyLwdRpjC6l8M0mxPgN2YFvTb87o4k8CStKj5fbPyCOYqRDPISIwOUA7_onr1UUn6vZDCAV
VeSCRfy8eAQ3Sn
"""  # 3050, SOTA=3050

X_G49 = """
LLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggg
gggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLL
LLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLgggggggggg
ggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLL
LLLQgggggggggggggggg
"""  # 6000, SOTA=6000
"""
ggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLL
LLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQgggggg
ggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLL
LLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggg
gggbLLLLLLLLLLLLLLLL
"""  # 6000, SOTA=6000

X_G50 = """
LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLgggggggggggggggggggg
LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL
ggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL
ggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL
gggggggggggggggggggg
"""  # 5880, SOTA=5880
"""
LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLgggggggggggggggggggg
LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLgggggggggggggggggggg
LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL
ggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL
gggggggggggggggggggg
"""  # 5880, SOTA=5880

X_G22 = """
3NbGRdQwo$UleJIHz3aimCPRiK5a9$y_7l3rCmgS6prQwKeXyJ2V9uJePA7WwL4_Eqx37mJaVUNE9V6qrXw1cr4Q0Ozv22bvkqee9QEAGcV5DsT0TNCcJ$RG
9wGK3$TxE4j6PYXgxdqIaXPGScKsPj3BvpxdNn3Wfy3tfL9H3zddbHofnQ0bMLX5AQEBRb5gki2YZ1kuwTlgc9l1p_qZfuSUvPf2DWx4nhMFYgQ3NleSc77S
XSSzTD9m6VMKrfbn8CbZGWtwsUkQXb3UW6JnnARtR5XaZrW$x4NQ52LiVrEZpFIQnzPsfv8utMCNptTsanvIvZQ0026wJG
"""  # 13359, SOTA=13359

X_G55 = '''
2VTaG16Jc4tj7oaVSD03sSCB6m5PSNJ8UxVzXdCBqpc7Tut1q55NV9g$mLtF7XO7qqKOHZpywPOpynWtkPosv9Kwrw9m2u9JdGCYgONFuworfD4$sIrDyUBp
wXNqalEWd7ygkwISznPscKYvsWc3GdpXig4YOeq91BFIGh2C9uDx9bKJt9uG6s2Iad9WcDMxcOEWrNu8uNJh7Jbkzw_p8pCZRFa16BJBLwQ2iqyB1W1YEqE3
qUoRf1tVuyk5D9NNDtvIVE_ScKm7FQqLJgAy97PiHxAkbB34o$Io67rOnOo5KmUbrL$EamIEk4VXV2ItMf74gc1gU7JukPofUWfbo_ihFDGSbcepjrk8tcMw
9Go_XsC169PLqugqzaK6A5eISQ0xBmzCDpzy00EP4dETLEJqcGqiU5TMQh1FB0FKgA31Llc4aag1DmG6ZDbkIyZtoOd6Khf3vPsxrBCtL$XptsYZhWwOPzxZ
oqXZbbkMFZ7CEb$YWt$K1UUvoGfYt2zQUki7iZ1x4IbU1nkwtSyPGigkLTic08Hep57encZ56VMjpnoiRo_JWhYQyq_j9yf8DF4zVuK0hwRQ8EtRnOTaRlzg
OuH235tBxiv87rIqmW_pWvhhLY1Jzqtm3FRWXl1WQ_EiOfl9qF8nWEjLl33UXvO5MByzxno6LFJdXmOZnWszaHoA3fznrvQssFHjB59K_icrHvp3Ytac2gKo
ONhb0xSZRpn$Hoz0jvQx_23IFRWIlGhWIqRdvudD3ikaEMzgqAlZFEPMUYe6X$2EAxhqZupAkCCeF4eHGJ6GrMnqj$DMme6d9220sT86G1VbRzK$iz
'''  # 10297, SOTA=10296

X_G70 = """
FBtWcG5X07seVYDuz2jQ72Kzrw3Oq5n1b9BfnKBzT0KO2PXMz6E$kGArGaiNwSXw3wrMNHXvy8y5FS$nkXVzzYhuLd6M6qZnHJxNJKWuIaiOCUD_KzLIe0aR
8qgkoKC72f2kOQVHvLEb34byr2nN3xkw6aL2QkpUg0sbI3Lbzu$wBWL2P_N2FTnGfXiLIOdSeEBtQSiMvmJhA3Bh5A4hO0wE13PltnYr6BHb6MfySrZpqViR
LKF7jnVlEXUP2HQsA0KPquFn3q5tSE4kFoBDNcGYuc0LvSNANGMudSQ1oTlRGt$V1RNVFzeBR52dsRiofnBxbfFUn4TW1glAqHXZFgAdwpuqcJC7M52m06Kr
AvFPo$x_sIjPIe8AVZeSin09YFZpXHTlozHxnuqR1vyz$DZdo9jXTLRO4AHmjZ8eg5ux3GdwpTRzPYCh$DbL7hO6qLd01AS1vkm2kkJZBI4tuBWIvYIkrxjC
T5xuFFAvAHTa7$jw_shWWHaCK0ls1tM4blCfxWHwW0WGCi8Lm7VstrVxF4IciO1gazimp0dS$XyyJ8O$2VCX5w5dv4qxQ7vmOws$sqfUosEV8MURKs2HIEqK
Qou63jNkOwUKI0_tMlBfGIR65uUTTlPSGMAKQefzNFMhaSuXUFin_Lq4d987P_lv17vQ46HuWcPIZKbrQ9KX0WdG_1nHoZuhrlpMVQTVQsi4Hu07lTrJ85Kc
sJEPu_EiQ15P9M5pDq5qb4uOW7UBEK555vYyfJ6CnuJGQWDDzeXN1Wqb2VBPeW5g$qiTs5jf2flMSkyHXekznUFQd0f9f4F_xxlqRtwSwSGPYUWZx$_wC4X8
IKRNNOphRCLy3hsUege4QBtkCXn9KPbA6VP9qEtRm2bp2p0jtTRXyyHmLL_nePcA0Yfzloh2NhW7D4X4K_7ELw45fnxLfb7PvsOAEvKPIl_XKpmK2uw373Of
XigZL8ql13e9T8NdfIREuJV87LZMovQv3kVnxZZMuV1_xxTP$0L7$O57N3W8sAtHhFpYX91UcTQww2bIGe6qKnii2_l0ytTwA0vNM59U5A32Fu_x$wNj4UGM
HorT1OofXde5kTO_$A3bKagyku$VT_L_boNivYFraZ9bY5Krzzkeb28LexhCpvGtdxKqvHSS2whEwBTnNQzmTJqLj$v0opnpUPfiMvz1iRfd9oyMUvZ_XUxX
AKDL38_nuXY5jJLXEFM7RUgHNLVEv3UtbtiP2E6b$GoHI4xT21b5MsfJcaQ9jXN522zUfS4xEgsiMQ8LYBInwP6Of69izJVskmki4pMK52mOi6rx5gBlvLuB
jAaImcY2CQXXvHwCPKTFgAbn_UEbEVyn5jcTN2ed_bA2CcHtw_a8iskfqbAq4RHx6BQNHeUoiiCAHY2mgXtxOFPgz$oEVO7tbJ21Nqt4L0bJaH0rfBf8V572
3ogtqSHkWz$RcJW5igq$A9CZ8u1KQyUMcFPr5JOY7Tg2feuFTM3uouaDf4VFs21HOHqZinUjG20adiUyqIPQCUx1NJJqDUauEOjD2NENiGuwDNxBnjDkqBO1
isopF5$xuJgqYdhYu0jIh9tWeN_QTLB7aUHArPGkqRqAz4X$RHLijCvwWUcA6MicAjuha0Ec7BZLQnL5HZWMSTzKSSNs8I0sISExFxu67L4
"""  # 9568, SOTA=9595


def check_solution_x():
    from graph_max_cut_simulator import load_graph, SimulatorGraphMaxCut
    graph_name = 'gset_14'

    graph = load_graph(graph_name=graph_name)
    simulator = SimulatorGraphMaxCut(graph=graph)

    x_str = X_G14
    num_nodes = simulator.num_nodes
    encoder = EncoderBase64(num_nodes=num_nodes)

    x = encoder.str_to_bool(x_str)
    vs = simulator.calculate_obj_values(xs=x[None, :])
    print(f"objective value  {vs[0].item():8.2f}  solution {x_str}")
