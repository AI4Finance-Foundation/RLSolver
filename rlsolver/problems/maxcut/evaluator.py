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

        if len(self.recorder1) == 0 or len(self.recorder2) == 0:
            return
        recorder1 = np.array(self.recorder1)
        recorder2 = np.array(self.recorder2)
        np.save(f"{self.save_dir}/recorder1.npy", recorder1)
        np.save(f"{self.save_dir}/recorder2.npy", recorder2)

        plt.plot(recorder1[:, 0], recorder1[:, 1], linestyle='-', label='real time')
        plt.plot(recorder2[:, 0], recorder2[:, 1], linestyle=':', label='back test')
        plt.scatter(recorder2[:, 0], recorder2[:, 1])

        plt.title(f"best_obj_value {self.best_v}")
        plt.axis('auto')
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
2VSaO16Jc5tj7YbROV8ZsS4R6m5PSMJDQRVzXdCBqJc7Du$XK55JVOglmHtF2Xw7qqqGXtsyw9OpSXWskTosv8Hwro9u3u9JrSFYuPNDu_wqeD4$sInDTIBZ
wWNqalEad7ykkQ2SrHPscKoPcWc9mjmWjg5YOuqP937QKh2C9uDJ1byHt9OG6qAJcd9WYCIxc8Ee$7u8vNJhMJbEzo_t0tEZRFK92lxFTwo2iayB1W12EqS3
qUYBf1dUuykJDPVN8xTIVCESaKm7FQqNpgByD7nkLxQkbP34o$IW6StOnQo5KYUrqL_E4mIAl4R1V2ItVj75gc9hUdJfkPofUAfbIRikFFKKb28n1rkes6qw
9GYzWsS961TLyuwyvaK6ADeGOQ0wBGzCDpzy20gP0dEPLEJaaGqiUDSIQx9FB07KkA55LBk4baeZDmGEZ4rFIqZtou26Mhf9vOcprBStL$XxtsYZZmwOHrxY
oqbX5b_IFpcCE4$gmttK3U_xomeYt2vRQkl7eZHx4mrU9nkgtVvPIiQiLTic0OIep53enkZ56TNjppoiQo_JWhYQyq_ePyhA4F0zRuK0hwxEAFMRmPTaRl$g
Oe1Y27pBxiv83r3qmevxWyhhNoHJzutn2FRWX$1aQykiSXk9mF8nWEjLj32UXf84MEyzxnw6LFJdXmmYnWszaHoA39znnnRosB1jBL9KzidrHTZB0tac2ArQ
ONgrOwS7xpnNnozGfvQx_2327RWIlGgWQKJdvuti3_kaFUzAGg$ZEERsCYe7X$AA6xxqYhnAkCAkF68HIN7GrMbqlxhMme4l9260sS82G5TbRzL$iz
'''  # 10298, SOTA=10296

X_G70 = """
5gXwIRKvLB05LYYX9scVmMZzDq362k5LduXt9$mSCZoc6eNg_rKEgbFOhJi_dDe38G1u2cpZsM4vhSzdijcVsXFT_mLiN5pOIqAfOzTPoaP_8GkJR446APTf
JSaXkSvIxVhcUB7c26cze9ze4uag7JgycspwV_prZrp_IwKkfPrbP8KFz5YsJgg97EgY2WZ_4j7gpD1Ax$ycOXzazq_e8fqKmJAHlubFNZRBlBU5vsNl1AiS
TnsWp8CVl4xNfaqD84qgIFSQf3XpZM2sE8Bi9oW23vQ9B0TASbBjJca18aoPXSi3N6m82lvZz6WaIRD0nylh7aqOtOj1lrZBKvlj1BSJDW383LYEkX26Zt_r
VuDuI$ZchT_xn9Yedh2oBN8K2CXujPqDWTGfh5PnyuQg6uT2fnXMRY0NQOLNYxOHQ0wljzbgVxlA$TL9lN9fofBV09Vi2zuB$yJMq4NQE8C5HjfsXrU9iHg6
DFVJk_9qujCWXlN3hD8OZXVKYaF_iiIwRpW4UxFGzcE1sRd3bb1Df_xIBZQuwGvUWVOrRVRufcMaJ$_NLdX2MFjxNA4BHBXLtFgYRALLAN$XFsTdUE_HtGBu
I356XIISHudo_giKlLAkcw1i_5KcdhVXrGQt0sWS3$V19gw1TQYOdL3jt0EfN7gITEWcu7Vaw7BEIZ$zNd23v7lsWUPAXrmif$dfLgpEwr_OSPSyAqiwXoAL
qMIwFJR5mipfTRTimCKXj0SJd97r8Qo3oAHkACEhlhIKqmuyrF7FTV253CkDvIRBRuNf1SX7mzV3u2FvDoxhEcYb2OudX$fQS8a4q6rQh_skBG24JRabm6_d
wI5ZLEhqGbo6gkovvAG5vQUq9RP4Ioob478_JTzneUavNeyyAdE7cuKsdpXfeEav3_NPvVUuvvxMa6xaqs4_Sv41jUiosnFGU7qW_AGPYy05gnviJ8rKG_5v
vroYNXHg0fglzYHwL5DslLMBhz0xNnxp295hzAHp7On7607VHh_YQxIv$Dc4lz4Xn2qXr7L6gqAJxAkQZwmb4dFMum$JQ1fgjS_DnOabbRfn9ai3QFKky$Gc
rHiaBMYneaBTjd92EQv_4aOOxSIzIEJVtNLPv2clXf8SB984lz4rMrgCEIx_sXOtqGSFScSemRmMiTJZQiZg3UXUxNhJiGvVa8eLwjyOC_8N6IsnUrxk8GRw
8Ja31nPwWsShEMpjr_6AZF2gXezxBgubdXh98CjtMTQCO4FL50oSwBP7tX6FmsLDBej$ke6fRCQgQW4XZox0kbdQ3fFy6w_87leq2qQ75hCQatL7zZPluwGH
QeEKijfDNsnZgFl5KSf$j_Kh25no$HUp0$K50Jkpha4eXfNV0yasGov3bQ5L5GPYGAaMLluqOMmthGgSenLan_cd90BRWGfXQJJj64h88KmIwjAXKrNmFxWw
YPWO1chl8zx20J7$gCmvASk4jH9pJKkX8RyHEH74cwS9pWjmywHbAmg7t54QwfcFvfLyzTrziJaz2oHyj03ypTZWc1__SuW8wfUXfvV1tU86DxLVnurcJfO1
hI1MHL$$n7W32E96659blS3WAnnGOr0Vwg7MMvyKS8ignmH_pfy7g1TeTVF1R7SSnUPCojEBO7Sz4ds6OcGu2QfLzCMcMg4SRJho4RueZxm
"""  # 9583, SOTA=9595


def check_solution_x():
    from simulator2 import load_graph, SimulatorGraphMaxCut
    graph_name = 'gset_14'

    graph = load_graph(graph_name=graph_name)
    simulator = SimulatorGraphMaxCut(graph=graph)

    x_str = X_G14
    num_nodes = simulator.num_nodes
    encoder = EncoderBase64(num_nodes=num_nodes)

    x = encoder.str_to_bool(x_str)
    vs = simulator.calculate_obj_values(xs=x[None, :])
    print(f"objective value  {vs[0].item():8.2f}  solution {x_str}")
