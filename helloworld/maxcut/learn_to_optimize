import os
import sys
import time
import math
import json
import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Bernoulli  # BinaryDist
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

# 变量名缩写约定
"""
edge: graph 里的edge
每条edge 由端点node0 指向 端点node1 。在无向图里，端点可以随意选取固定顺序
n0: index of 左端点node0 
n1: index of 右端点node1
dt: distance 这条edge 的长度，就是node0 和 node1 的距离

最大割问题 将node 分为两个集合，集合set0 和 集合set1
p0: 端点node0 属于set0 的概率, (1-p0): 端点node0 属于set1 的概率
p1: 端点node1 属于set0 的概率, (1-p1): 端点node1 属于set1 的概率

prob: probability 按node0 的顺序，标记每个节点的概率。是GraphMaxCut 问题的解的概率形式
sln_x: solution_x 按node0 的顺序，标记每个节点所属的集合。 是GraphMaxCut 问题的解的二进制表示
"""

TEN = th.Tensor
INT = th.IntTensor
DataDir = './data'

SlnX10_21 = "7E"  # 16, 60sec
SlnX50_176 = "0OSV_ysKG"  # 135, 120sec
SlnX100_460 = "4pcoGEDyanPbZyf3q"  # 333, 300sec
SlnX300_2036 = "0xSEMFgaOogyQX4pDxVsvcdJftoSGXaWF56NBQASMFabooHBENh"  # 1410, 600sec
SlnX500_3624 = "01XMcqyO3LodBuHYkrnZ1AKycmGUKj_LECiLo$QtoqK8Xziw$$Oqyt6mqPuT31Wau1s_QEv58HDAZYdNZSnC"  # 2474, 1000sec
SlnX700_4036 = """
A_2AwQpd$iJ2kupuKHx1DHR5YpS_uhFAgy09Bd6LZpqiJXrqIJfkOTpAyg4$216r9Tpn6Sdrgzmsv1XV12WejbhAyBcKlNnpN9aEa1nOGTR41pTjhxakO
"""  # 2828, 3000sec
SlnX900_5122 = """
WuaXXJBXvF5SVcqUQgnwCgtqfjSCeGAhfcpUQnXzBaWUkp0PSTjspksGsKO21rxIIKyPPfXrIvrrqo42c0N3LXks3PiMOdNn2taGyQk9zfi8JdU8UiFwZbgl
pKirUEh3aTlj50bFnOEKaWbar4Di7w
"""  # 3639, 10000sec
SlnX1000_6368 = """
4Rw9snVzzAatq06drjlc4sKFk2h2DwGLGORocZqJ7hGo8U$gLizM7O1l4hBZKuqb_WOyRV5VvDSUZ_X1et8zGEXfC0uFQUCnfqBvQWRn6_j2DXL8Y_9HwyHH
76vjSpS24S06DSNJR5QDAUiBU8V_fRix2LoTBDl2lgraDJf
"""  # 3585, 20000sec
SlnX3000_25695 = """
2bxRw7$qiA8qa1wB3c7SjKXwqRcUwjLD_cDjSUG3Y0BYeVcEgFi5GDUxhnr3ttKaWLhqaJQ41QI1NWu2wwfn_l5NRyzFPA5m1TTefLE47bPtW5tfkw9oeIGB
Uiuw20ktZNicS$$At3QyCMaQtZuZ4bC64j1sRYbly2tDcZlFljSzGAuRS8swdhOdFQIB4I2acx447KcZl9r$30E2rxqDkONNwFhIykWgiHbsCNADH96a7eo5
Kx7kf5JAI7IhEU74sPnF0yVmfWY0Nn8foalzctO3XYMBXKbdAqmvCEo0cwf3eiGH2dnJ7JN_d_dOgrfsPbT1EhTAOaTX69zRDHuIOKcR5WAoFxr9pmCcJei5
bK5Kg3tL$2jRpBdPblV0_k1ivLleDE0g4v87VrQa0rzU5KRKt8$8OquVcCFGqzePDDEvZElacSz2mON33r0$OyN2cBOWKI$FpdrJfsJczE2iHQw6i23PbNgb
Y3UOud81KwlRLTiUG0nK
"""  # 17176， 30000sec
SlnX5000_50543 = """
1QTuToko1RdZNMKROw4AIZR5I5ZfRRmmBtWi6iqzKxIThtTHaUuhYcDXwFsReQuccxBszkvZsg7X16EXJA1N7sLnaC3jmRGEFjl7VtHJUOylOksOW9jstMTS
5ndjZivBfHtMDUUP9izWE6VE__lnGgA3R$_tRZAW1hPQuHFnVIZ3QjZA60mqrTP42MJuhDdgNbbP4$K4_eD7Lp3n827f3G$o4MoQg9BFXhC$v8ejx_m6gktj
r1PAb8MqAwFpg4azJQUSKejhfUjCgDicpkJY17x3fE2_OVhWhrnA2i3MtudH$OC6SdOBkB9fgj53_Ka1CclMA$MSlUI8YN27mplvauk29BTEl5fjlXyUfdWL
OYvdtBY6HH1p33C3UN3tG4jsdiTwE9tXW9CDJZwSvS3ndr9weBiiHAcY7wBLyNkSo$z3AJXmKGx7kymAF03RjVH9_C60Nh9ByAZv9n5cDsH4yOB7zeTlslvX
jy4peS1bvF_lYVURjiAZ2nrwldRI_LL_pnlqmN9oj5bO5kfbQEXrhm44CByNoJ3cPOMZ7A5Xj3bteF7lNPoqzLu7OiR4hXBHFB6dKXizvz5dTPik1ei1fSbw
WvdRGhadtDZ_wMHWA8r0YhXaGOBZwctsJC_MQd4LdMbOglmiQdhyGLSz_hDX43_MiO_KwjYgwXVYmaJAeX5qS6xYmWZyT0ZI9eu49kAF3RWr3XQiUa2bSLRX
zXmCEr004t9DyW9ukZzv00P7ogWxRflL69q2f1QI9XJgEI46WrHAAulbSaucMZY1$fbup40FL7V8qmqzhNnaPhj0LCXC6_eU9tEGQk$$Fhewl3TuJI
"""  # 33217, 40000sec
SlnX7000_79325 = """
F5QNa5OkZdpVVcftbBXFD9fNitOCDfw0RP$doUdUc3wGry8pJ__jOxYoMAj0oBFJbX6jCzF0N7GgZmyj1oFFggv5d52VZ0Tsq4_aORNPvM8Hexm380$vo6g2
ZEfz4U9fNLG9_jLAyKNO21aGEB8Crgvs3mIsjr7bCDmhctp8vkNSfrkn$pm8fYT4zDWKIxqDNDWLqDbcWNTl$0Mgywhbs1HM34$Xj0cWdcl7GOnNNVnbkGqk
BGDMim8UNT9sTRVRmPaovhhAh24NEIiSo0_hFkjanbhMZcKcfMhjWX259KdkpNeWLNOnG8xXxaLiHQ9VTI8xd$IcMI58qY7oaCCzm70E66K09T9$zz35xA4b
bKlp3OpxeDjxEN1$ywfMgQiEzH8$t9JWTPetKLEeKmXxW_adxpUu7QajU8QOPeb4ff2_LU4Hz61PJiIv_SFENF7UhM7e9wqX7axxqDKchCo47IW0GM40zhGb
KcG3hJdlVhpkQ4U3ZOsC0UzzKMr5Fr4qSHg0ofrIQvFyEXRA7QTWO2ZNDXlV4C39D8rsAD6YwLR9GI$8HmY4Uu3$zVJLguELDBPBgQH0fsD2YPi$PzEuMPx8
YrxLmmg$8v$LgwfMIa9uo_1HbgPxeU7$a62oJ4I4G4kuurzbNmd2GpL1s7xJe8sA7pTeVpkRRNV9o6k19GIBYxv4GL4lOd0lfdz5sBMWYFbpHmPextPsXwm_
YasGgSXtP$zij3AvNi$yvJN3xRdZ8toOmE6X_UIcCh9aar61au5Fx9h1szb8xMFN$r6JV0t6Rg30a3A0pcPOoBD_QMBKzywCncszhYSz2I_fUykvoEumcCIv
7prrmIXETxofF07I_qtpN2jbCXn19g4ONPi3WOWAT9CEEm1Eqg7mTmMMqcgsz_gQL7OA7FaYM68xRjUPDpaktuHSTVB6zqv29ZcvDsAr_wNlazCyxodyACkG
a0P_IMsZmHjQqJcJfMrYmo79PobXbOTPJRn2TfbuUOZ4sJglrRIMb7g9lJh9DNOL$Ztz0_BGos8MEdUFFoMRrKscvaXcUrQenLwuATKsEKYCREeTpOail8qx
UETF3NMJYhkXYfZx9jBpa5tGBLBpwb6FiahM92Zmz33B04s0rCB6BPb4GC_MnOV0xGjy4l6un0kUg1jgo5ZFqvS
"""  # 51509, 50000sec
SlnX9000_96324 = """
5TouSYYQQviol1nWgywNJlojouxYnBZ0jZmuc86rmcFhKiVWxj_2VAioa57XUUuPIx$yt_WITVFKS1cZdp_QKchGwepYY4F2h4DCeW3Q2$6_k8SltCCR4g_y
dr4VCaDo3iPrMUt5PcjqxoWtGhWInykE91eg3G_5nF82P6v97dgBU$2ZCsgn8J2mCtQgfi6$7gc9DWCRjLefYpwhEDLobTyqdRmNi8GdXOj2TdW50fUo7KjX
gTETF58UTfI0CWIJDNuZnmcbAC3RQ7uLToi$q4luh5RRx2STf_kua5U6PWcMDVqTljemD9NNLCy7oJjySosHjYw1R66UQIqrL50_yyRHQnvoRxVTZILYd_mF
jF6imCzPSHsWr6byh1hFkHd9V7v7CpwixGfUrDigaL$C0_uxzg1sgM0hP3VMbxhN8NHmKORKaMQ_K$$x9buOP8zYDDADWhC6dWSPLRBsWcaPVxhS3sp7hKgx
kGyseNQzbxLFXhZpGJjjSQ_kWgluZR5rEJP3CgPvUrlNLWmf$PoIQ8J3T429CY5S7_SNzYP__AKw09d_Q6rjbV2lJZlZGkXEc9tBvytmIWJUTNu6aUR3Qxx6
yTHbZSIpEBlrDJZM5nxW6XKhVqwDr61mZM7nc3rbMIcIuQNAZblMgjMo9GhCpo_tKS91o4tdinCfQUSwmoaveymgrbtws9mfyLcCBDFJ9XlbqnXD6rT8ucsF
Ie7BDrWrjNVqMQp_kmHqerTg0hpNS5z4Y_DZ_Mho28GUJlXyB6u5bpGfpkRrEuG$vN8VAmrzWoerhVuBEo9J7aoKKEntl819svD_ti5IJpEHeXqNA8mW7YRG
ze$saXOQRBrKd0w7dyvQVpmniiBsYtZiSQ37JPuIiZvPJ2KkROBxLL44ofK2a20Tm5KlZC2bvFz9MPJps05iyyJ6oMOtPaAskxadPklQ4C4LV5Ip1MiZ1PjI
tzwm0PQExU13CGPZW2ks6glHREI5ZEmeuZBkfeG$Ggfp0_XxD5KT4HxkxgoaAh5FX2E9nfI2tn3gM6hrx89B2ISYqH5vctASOTPZU_1J2M1Q7WL7$7FWKqgJ
e_JX9XL6J2uNxKEfAJBiFnVrT5XxXKokzOg5WYAyXcWQ7y_lUtpF15zv9DcIge77iRO9Ohhwz12vO8lFLnyxcP2Q2Ri5IYrTd8SmH09ST3pzuUeEUyzzAaVS
mBYiepI2v1hegq3ObZjGImZ2wP86kYgZ793TyuW1NWIPHSxyYjjrDiSBBPH9ItlDludkXYrTtFsSfZ2o6W5mU$w0W8YQ2rbczrh6f3mfMQuW8YIKL5MTiYvd
nN$1LT$T1vsfMrJulb33DsxE1fuR0WgWkV6icEDTzLP69CcAkbsdkTZ52nNwLx6eoRpXkj3yBVm6gvlyCcPLR7zKV9nuFypsngnscqzg3UuGScu2$56CnofR
GKcdRBAag_a6jvH4IKfpfUCQPrbMaTSVC9gOgdAuzjgJJEeEtBCnuVtSjCbP
"""  # 62809, 60000sec
SlnX10000_100457 = """
227U3lIPCygvU9Iq$2v2p6Qr73Tb6z4ViAS5DY7mo$NHGDbK5iw2pY1lVnzRJCGJTZ9NbTMaJ9vT09LEFog47rd_MMRCqn1UHGFbK9Zfp3m0vyVFiHR$uYqF
ZNwQ97gcPCF1dMQoZSqw0gwqD83mz1ree6zEpbynbQRdH5oRvrerbAkXsMIEbSjKClWPkvZhzTMqsR$0eKTuJlwGzHJtDaJScBStx_uJXwt2lA4io4jFVXZi
9M3teQmDf0KDQ9hoGaOBOMFB6KTk40BMvt2hVDv6Ve00WCBvH2mFfYiXFMTXzhnaMMqB4aLTuEx1IR5QeHGffOI$iwI1CXoOwhm1HnTmvudGcwj7fxzEWl8$
214yCANA64e_Upg1IBIhyo2VvO9WBRlz5lh4tnnz71WtuZGj6S7CsUbVlxRA8yw1BestsRPqkmlV8IWBA8Mx_wu9LqgZ0PZOPT4p1zm_h9F9XWbOPUj5jPM0
7SScaFmflrTKJD58QN7occxNAQpqvMlX_YThqpXRmVVHNubjhRvYPeAgQ0PGjjdaL28Hy8OBVVVo5Xgh84R5C2avrw0zKWtz5rovdql8kXhofApeHkuVh6N_
jLr1WNgTegNAsnT64d4$5HXGRqQzelGLeAgsATbKz_ceV2G6GpYnvVUodNEeVY1yLvmX9grtkvh5wzWa2oQJh8iIHm70h23UQAbGUgYNvTs5JGbShNWm5Ts1
BI7POS6dU561PUiOMlGucObEqQKH3478LCGdVAsdrj1Enex5HovmoPtmDmj3BnJyUQwPeNso2ZJrQpKsKxUndmFM$DmEvdI2Gwz0$ujd1onP1E6XXNbpR112
tMidUnxAQqm6KNlEr4zNtnGBg5_l9jmrOsyHmzH64Xwr6q6I8V0CTE3cWs_9iOHCNf3_KijTlfeNOevh75hwzS84vcH$gn69cN2z2sJMCx_owEwFRwnFzq4T
tKQvZrjpSWagBs$sBtd0QHloXjfPTzGQGJdqOnKZXD2ylZZXTzUyf62Viup5Wu2Z5XnDDi8N0JVozkMWnRAYYDSnESJFK8V448DrVzoPGvWaEo2YQrsJvwaN
lPRE3FUif67EeJ77YTTY9ewpjts$RVEER9sXb_xKMBrCf_RD4Og0nKr7Bd0WpnirFJklsBhzaaX8koF_NL4EwHkiadWqC7dYI2vfhIsMbGUkwjJXNgOZ1Tdl
D2WasuhiejhIoxAX7NgP6xl5C4wxzZQr7Zdg6EfjhyHd6evOaA$IqV$WDA0gg0Od7lxQPEF92d1HOAt8rolJ_t1G3CtUwwSW4Xh1oMSM2JdIMic9c1SlZyJX
a3A2Ih0V_jK4f2noWiMRD_WIUFu7MRgeDixXhhfPjvLcochWdiSiW1eGbPZRqaeZKkSVAtRORsYaMsTZp$YaY_MsPoTDzWk5qEcHQjwNs52bQt3Jc0mumFVB
_1xkWKMjNjIVb9Y$yWTtXI1G5_wRUZ721rkyxuyDYmwcJZdpgTYEjd9ISi84f28sq1DpvYlb2n8fB_LllMJS5NBkCEbrd0pTPrBaC_z5l36Y7xtXCAVDHRCo
cXcLyJhPozOAayWMlcaKQiC68Ov09Sn9wIMfegfSII1n$C62Zl93hiheN58gSaDVMR2nqbiMXNnRd0y$sKyqAyypQ74euNrlEyQ6PSzDTf6
"""  # 66109, 60000sec

SlnX14 = """
2dbChJAXfdo2GRp49ecgPjQwfRSIJqcfANlGZMwwAOZzjFXMXFcLYmRi27fT49J38CH8NUHFf8nLFzuUQh_LltgK6ofnt0P2NEwwUdURMPtFC8ZhlzftdQJj
MQ1aEyvV6RqIv8
"""  # 3029
SlnX15 = """
AqOThDGdbjuzr7FAoXiCBgbwlc3lsy9bo2vb$KWBJ51OOIjkGcCBKwsQtu0zrlerpyWzpoAdOyeeoSMb3SoG$DSge$TNnfvsDwiivHk3JNFWTdXtiea2IEAg
$hxSFRWpKuqFo
"""  # 2995
SlnX22 = """
15M1wnHOEdRzp8Zws3XH0KzUDip$CCuC0dhgj9onOtyhWEbpH6hTzmSNozsdeNqM6jW7mwIC_nfrh1TYS8uz8St1$gz4rvd$8H5IfzhOOg3bXg7VS$4GtxyJ
$YothXUyFlOmUzaRwD31BeyYHKZPejOTcvXfcXsL$m$utCAABFN77HQCYHq5eNgTY9hy8oSivuLd09Crlz5AKI7u7ZNBKZPtOEM$sA00vFU9QmkX7e6JzPS3
32fItphr4PhDkGBFADHgVNBREHqPMO5BeIYPAah8chsT2ZesqvwFF2PE0zUbyNJY$2iYA1RCZWVxbPo1CqfgQaE5ddDWrF
"""  # 13167
SlnX49 = """
gggggrL9LLLLLggL9LLLLKgjQggggfLMhQggggjLAbLLLLQgbQLLLLLAhLgggggLLhMgggghLQhLLLLLgjQbLLLLIgLLAggggbALggggggggggrLLLLMfLLL
LLLLLLKgggggbQgggggggggfLLLLLgLLLLLLLLLLQgggggsfKggggggggrLLLLLggjLLLLLLLKggggggLLAgggggggfLLLLLMghLLLLLLLLQgggggbLIgggg
ggggrLLLLLggrLArLLLLLgggggfLIghLKggLghLLLLLQhLLQgjLIhLIgggggLIggLLQgrKgrLLLLLhLLMggLMgjKggggfQgggjLMgbLQjLLLLQLLLLIgjKgg
sgggggMggggfLQjLLLLLLLIbLLLLQgrAgggggggrgggggLKfLLLLLLRKfLLLLMgfQggggghMjQggggbLQLLLLLLQbALLLLLAgsgggggfLgMgggghLKbLLLLL
AfKbLLLLIgjgggggjLQj
"""  # 5712
SlnX50 = """
1asl2rp1eUku4M1dd8tQp2xCYq73qJv2Ll$zNZ4zIz70GXWWyrK50DnhW7HivZOs9o7Kpn3G$z0HmouyurVAiCrlvuQQIGaOpDDYaQUEZkr0vvHJH8aT8srZ
f4rKuOJLSTL_PmfKRILzDVbvy1XUR8HeOUMgGgQfKsBgwz2onKgeeCV771nXRt5RktBFfPRVU$IEQHDMHdLsMEtjq4mMPVxc7uMNVUYMcSw6I089eEkx1ykH
6e3_jRGBhH4a6qth_S0ZIv$UwEuH8wscQ6Q$nqTuwUOb5_Ydgkk6OjTY95sE$qzo5D1sChcow8O20xNR4RI9Cr6ac5XMiO5YSo9xq_75hBPe4WhPCRqisRb5
H4hdkKDeR3ma88A4g_$3y3psPhjRg0IoAO9SmiUUidU50IFae_uiH059oguuBiWxXwq0JbqNQ8L3Bro3Ct7IoYjzxu3ItoFE6JnJveZamnZOkm$fqqcTUzXo
4qo_vnbZNzCWOk8LB7_O4kUDk1FnpyslVD2cTMEzrLA0V594XUMuG6mal_QWUxykxjGxaOYeHrbtImmGOUQwQtlMbibAjpbWNHPKZcS39fPQik__1wRBM35K
fvMW9vdomQ8QldRyh0dFnJmFCv0DZoJgXMUb$zALZzt9tDp8zmkelaJ1z9phwAefJCCd0ZtkyGDu8HaVnLbS6LKO$e7uTOrogXXmW34uQRKtsBTO2qUj3$BC
S05asJATBxRzlynBxsX5_6lHTiS4zqEw60vyCl$8njoaEFEqq7n2bJI81gJkwNliIRdWFkNAC5lqFUlSo7$3q7DiY1zDVoEkyMYsI6wXzHb3wEcF_U
"""  # 10015
SlnX55 = """
5ja8A71xNG14nx9epezOKNbEYn5KxW3g9kvmtnFV_0XUmF91nLqJPZBxR1xMDlbhyIRJeHlS92mv56t1iRMRKuiw$U2wjG2vbAIeV54Y$WeE6WO_npEwpTFp
mscK4S6LLeDnwiGwusxZmu7LDWK5kLjcGkYnDwzKR2_O0bdmg$JDfa15e0Y_ZbRlm_GduCzEV4UEO85CBQaDh0iuRaoYtS5X7wF8SAJ0Ud4tW_PM2hVLb4fn
jcR_bXJrounRgfCCseYQ$ZEh8v3sMlVng9jp4qxQyCRjzjkciAW7Scop2cEToirYLA9KSt81sWJKQtNRueq3kqyfoNZcJu1_HOZjcd98UhVztCAivIeaYskk
J9aCyHdFB4GXzFMXoP8t8cplLYoOG20$9ZbsR6$CKoS9K6tK4ZaHl8DHUKS7jXmFGgdPZFyKK8_Vn0j7zM_oHpg__ojZGdDCVI_LZ0qHV7uLyVf4Z1qI3$ZN
cJwuR_WXrL3_UuOsIR6OUkxdHZg26EGtelQ$bnhnKSM3BteaMuFWqwtRfCzqxhktp6lspWOPna$z$s1M6zeWsZFeCM1HZnWLmeZCjy2debl96Q9yl$WHfkoP
pabJLRSBa8dc4K3rflUIrZ3euDPBYH_HusXSgRv06CAFDV9_5zOcgnsmKNzXvbvrKtCs5WNrvTcB5qjOtlgDJmM7BTv_qEXgZsO3dloy73ZO7s$zybADSWeW
$TvQL97vO7St2jzowhxP2hjK1kdwSkS8d$4rmVvBQal3BDkuJqJt3y4FZyMzPPkEBpvuwuRcXe0Jhg0sBVt6QFuQd2SxCWiIuY$uCajm2R7RwNOKP
"""  # 10031
SlnX70 = """
3AcGoeUhKdVU72ZiLyfxgUIhgNQQdEfI569nHq$V5IB8ZsZiZY63iDDSZ6VJsToTMwXTCurzbeCHfyYL_CfRGW1vej_thDlqh$Ee7R5bcXVLISB_MiL6dOyJ
KWhBoU1YXPKuqol2GvULXW105Mh7XJEnQZXOh_NUfbRU2dtM$BRWf17EBl3VyhzOO_yGxi5YRd$dJihU5rRvYLhghN4jyEQZVEp$sxv$5w7vLHfyUrx3u4jO
knlNNiBq9B5p3opgCGog8w38H9SNoUZDjc$XDAit$A8MpKF3ouZqtS0bs46RaY7ZetlJJtlnUC5YKSeafaApJD$mY54ouIdSmH3pl30dsn9_Z3c0XdJ_$60Y
Hm3lf1Z$cYXDSCd6UlcqWGwPkAX7412M_i77J3YopwgsgvlJeyfGPDn84UmP1U9420mp3GiCGj1UupE7ccZc4hTczRs41FGvmi3lcu5IJYmpBAYQYmX0Thqz
PqSQT06_eLTSamhfCwFya11kc4hh1aUbbcpiv5XQqdWf8SvDt6UOtN__mYniq5g5YywE05WBWzio6bRZUr0mfJX7p6Nn34AcQQ_zbSLVcsZLENUyM29RKfkw
E1b43g04$mq0BgeZwVM5HMtt3mI$Q15CMOEriM19KD3I$Wx48af6oJqyN69UpyYy1uSuP64L0Rom3DdVOrDAw7uSFOt0WNus_$ZvSQzVEFLdQb0d1yNC9Lu7
AZWerkE$JrhU1Hfa2W$WyGwu6bpT_a3ZKsBEhUH6GCA3P45DrYrd57K8z$lE88iz1Se9ICPg2sxN9ignqQyjvSlKWQWe_eXSigDOBcOQZC9DX_iulwo2aI5A
uMRrvdjhmjef8EMJZFhkLp2u5gv1Gx3ISl7A5rxd5NvQ372XXJTYi0ZYXr8mAVRO826_tIs27Akoar1Kjy2SjRlrK1zCSq2XroIYV82voeoGGZubHjp1A2Ov
eS_kpXyV5cWCNhVomaD$knBhbL7Y6lPf5UkLfiz48PquooOh7aL6lsa3d7c0cwoRksn6h8RGcT9vvKDc01aIP5q6GEgAR7Suey_iE2HMc2Fv_Zr4ZYrm3MTu
4YHS1DN9g6GJsDCvwsp9LqYayT2wQA$H54I4edj7LrFb7ZlJjbVF57StGESAggUnldR6vsGvUkyRMHHjp3l9f29oIzf6QKztQHTj59uEqFDoypgm4ZvMlFub
z1rHQ4WSytr3OnZta42Au5Y7xatFdbl74dtyMJlXwVM9nNypB6VR_6fAkf_8phP7AtjEXzo0_4tkN59LphlPaQq2pLRqsNxskI9z2Q5TdoFmeb1xWkICPS4o
i5AB4LozUwz80VpPo4LDtVGIhs6jYME31mO0k6FoVh2i8gLRmL0wvqsBncNiaRNDahR5z1xuuGSERZQW1Gh98J7PBsYeVGsejp316q7yTPWR7nBnPQinyv1d
ULnnlmDdG8xSq6g1slTFR16ngzfCOgMNRdjbmpo8O1a9jpIJ8nrCYxiv$cFNedJJ3eRktLMXQ8QvqG_b7InRUfpmyYEFeR8bCvZTg7wZ_RgRRtP2$D3jApeD
sVKJ0hVuqQRNI3bWuEZNVPC_BhuQjsTMrKSQSH6YZCp7d5uRBpJbHI$fQT7s8RCkW5zA6GcY0Qt3YJS5j8mH918L28Os9$rK9MYIVlAMd7_
"""  # 9358

"""Graph Max Cut Env"""


class MCMCSim:  # Markov Chain Monte Carlo Simulator
    def __init__(self, graph_name: str = 'gset_14', gpu_id: int = -1):
        device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
        int_type = th.int32
        self.device = device
        self.int_type = int_type

        # 从txt文件里，读取graph
        txt_path = f"{DataDir}/{graph_name}.txt"
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
        num_nodes, num_edges = lines[0]
        map_edge_to_n0_n1_dt = [(n0 - 1, n1 - 1, dt) for n0, n1, dt in lines[1:]]  # node_id “从1开始”改为“从0开始”

        # 建立邻接矩阵，不预先保存索引的邻接矩阵不适合GPU并行
        '''
        例如，无向图里：
        - 节点0连接了节点1
        - 节点0连接了节点2
        - 节点2连接了节点3

        用邻接阶矩阵Ary的上三角表示这个无向图：
          0 1 2 3
        0 F T T F
        1 _ F F F
        2 _ _ F T
        3 _ _ _ F

        其中：    
        - Ary[0,1]=True
        - Ary[0,2]=True
        - Ary[2,3]=True
        - 其余为False
        '''
        adjacency_matrix = th.empty((num_nodes, num_nodes), dtype=th.float32, device=device)
        adjacency_matrix[:] = -1  # 选用-1而非0表示表示两个node之间没有edge相连，避免两个节点的距离为0时出现冲突
        for n0, n1, dt in map_edge_to_n0_n1_dt:
            adjacency_matrix[n0, n1] = dt
        assert num_nodes == adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
        assert num_edges == (adjacency_matrix != -1).sum()
        self.adjacency_matrix = adjacency_matrix

        # 建立二维列表n0_to_n1s 表示这个图，
        """
        用二维列表list2d表示这个图：
        [
            [1, 2], 
            [], 
            [3],
            [],
        ]
        其中：
        - list2d[0] = [1, 2]
        - list2d[2] = [3]

        对于稀疏的矩阵，可以直接记录每条边两端节点的序号，用shape=(2,N)的二维列表 表示这个图：
        0, 1
        0, 2
        2, 3
        如果条边的长度为1，那么表示为shape=(2,N)的二维列表，并在第一行，写上 4个节点，3条边的信息，帮助重建这个图，然后保存在txt里：
        4, 3
        0, 1, 1
        0, 2, 1
        2, 3, 1
        """
        n0_to_n1s = [[] for _ in range(num_nodes)]  # 将 node0_id 映射到 node1_id
        n0_to_dts = [[] for _ in range(num_nodes)]  # 将 mode0_id 映射到 node1_id 与 node0_id 的距离
        for n0, n1, dist in map_edge_to_n0_n1_dt:
            n0_to_n1s[n0].append(n1)
            n0_to_dts[n0].append(dist)
        n0_to_n1s = [th.tensor(node1s, dtype=int_type, device=device) for node1s in n0_to_n1s]
        n0_to_dts = [th.tensor(node1s, dtype=int_type, device=device) for node1s in n0_to_dts]
        assert num_nodes == len(n0_to_n1s)
        assert num_nodes == len(n0_to_dts)
        assert num_edges == sum([len(n0_to_n1) for n0_to_n1 in n0_to_n1s])
        assert num_edges == sum([len(n0_to_dt) for n0_to_dt in n0_to_dts])
        self.num_nodes = len(n0_to_n1s)
        self.num_edges = sum([len(n0_to_n1) for n0_to_n1 in n0_to_n1s])

        # 根据二维列表n0_to_n1s 建立基于edge 的node0 node1 的索引，用于高效计算
        """
        在K个子环境里，需要对N个点进行索引去计算计算GraphMaxCut距离：
        - 建立邻接矩阵的方法，计算GraphMaxCut距离时，需要索引K*N次
        - 下面这种方法直接保存并行索引信息，仅需要索引1次

        为了用GPU加速计算，可以用两个固定长度的张量记录端点序号，再用两个固定长度的张量记录端点信息。去表示这个图：
        我们直接将每条edge两端的端点称为：左端点node0 和 右端点node1 （在无向图里，左右端点可以随意命名）
        node0_id   [0, 0, 2]  # 依次保存三条边的node0，用于索引
        node0_prob [p, p, p]  # 依次根据索引得到node0 的概率，用于计算
        node1_id   [1, 2, 3]  # 依次保存三条边的node1，用于索引
        node1_prob [p, p, p]  # 依次根据索引得到node1 的概率，用于计算

        env_id     [0, 1, 2, ..., num_envs]  # 保存了并行维度的索引信息
        """
        n0_ids = []
        n1_ids = []
        for i, n1s in enumerate(n0_to_n1s):
            n0_ids.extend([i, ] * n1s.shape[0])
            n1_ids.extend(n1s)
        self.n0_ids = th.tensor(n0_ids, dtype=int_type, device=device).unsqueeze(0)
        self.n1_ids = th.tensor(n1_ids, dtype=int_type, device=device).unsqueeze(0)
        self.env_is = th.zeros(self.num_edges, dtype=int_type, device=device).unsqueeze(0)

        # 将问题的解这个二进制张量，用base64编码为str
        self.base_digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
        self.base_num = len(self.base_digits)

    def get_objectives_using_for_loop(self, probs: TEN) -> TEN:  # 使用for循环重复查找索引，不适合GPU并行
        assert probs.shape[-1] == self.num_nodes
        num_envs = probs.shape[0]

        sum_dts = []
        for env_i in range(num_envs):  # 逐个访问子环境
            p0 = probs[env_i]

            n0_to_p1 = []
            for n0 in range(self.num_nodes):  # 逐个访问节点
                n1s = th.where(self.adjacency_matrix[n0] != -1)[0]  # 根据邻接矩阵，找出与node0 相连的多个节点的索引
                p1 = p0[n1s]  # 根据索引找出node1 属于集合的概率
                n0_to_p1.append(p1)

            sum_dt = []
            for _p0, _p1 in zip(p0, n0_to_p1):
                # `_p0 * (1-_p1)` node_0 属于这个集合 且 node1 属于那个集合的概率
                # `_p1 * (1-_p0)` node_1 属于这个集合 且 node0 属于那个集合的概率
                # dt = _p0 * (1-_p1) + _p1 * (1-_p0)  # 等价于以下一行代码，相加计算出了这条边两端的节点分别属于两个集合的概率
                dt = _p0 + _p1 - 2 * _p0 * _p1
                # 此计算只能算出的局部梯度，与全局梯度有差别，未考虑无向图里节点间的复杂关系，需要能跳出局部最优的求解器
                sum_dt.append(dt.sum(dim=0))
            sum_dt = th.stack(sum_dt).sum(dim=-1)  # 求和得到这个子环境的 objective
            sum_dts.append(sum_dt)
        sum_dts = th.hstack(sum_dts)  # 堆叠结果，得到 num_envs 个子环境的 objective
        return -sum_dts

    def get_objectives(self, probs: TEN):
        p0s, p1s = self.get_p0s_p1s(probs)
        return -(p0s + p1s - 2 * p0s * p1s).sum(1)

    def get_scores(self, probs: INT) -> INT:
        p0s, p1s = self.get_p0s_p1s(probs)
        return (p0s ^ p1s).sum(1)

    def get_p0s_p1s(self, probs: TEN) -> (TEN, TEN):
        num_envs = probs.shape[0]
        if num_envs != self.env_is.shape[0]:
            self.n0_ids = self.n0_ids[0].repeat(num_envs, 1)
            self.n1_ids = self.n1_ids[0].repeat(num_envs, 1)
            self.env_is = self.env_is[0:1] + th.arange(num_envs, device=self.device).unsqueeze(1)

        p0s = probs[self.env_is, self.n0_ids]
        p1s = probs[self.env_is, self.n1_ids]
        return p0s, p1s

    def get_rand_probs(self, num_envs: int) -> TEN:
        return th.rand((num_envs, self.num_nodes), dtype=th.float32, device=self.device)

    @staticmethod
    def prob_to_bool(p0s, thresh=0.5):
        return p0s > thresh

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

        x_str = '\n'.join([x_str[i:i + 120] for i in range(0, len(x_str), 120)])
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

        return self.int_to_bool(x_int)

    def int_to_bool(self, x_int: int) -> TEN:
        x_bin: str = bin(x_int)[2:]
        x_bool = th.zeros(self.num_nodes, dtype=th.int8)
        x_bool[-len(x_bin):] = th.tensor([int(i) for i in x_bin], dtype=th.int8)
        return x_bool


def draw_adjacency_matrix():
    graph_name = 'syn_20_42'
    env = MCMCSim(graph_name=graph_name)
    ary = (env.adjacency_matrix != -1).to(th.int).data.cpu().numpy()

    d0 = d1 = ary.shape[0]
    plt.imshow(1 - ary[:, ::-1], cmap='hot', interpolation='nearest', extent=[0, d1, 0, d0])
    plt.gca().set_xticks(np.arange(0, d1, 1))
    plt.gca().set_yticks(np.arange(0, d0, 1))
    plt.grid(True, color='grey', linewidth=1)
    plt.title('black denotes connect')
    plt.show()


def check_env():
    th.manual_seed(0)
    num_envs = 6

    env = MCMCSim(graph_name='gset_14')
    probs = env.get_rand_probs(num_envs=num_envs)
    print(env.get_objectives(probs))

    best_score = -th.inf
    for _ in range(8):
        probs = env.get_rand_probs(num_envs=num_envs)
        sln_xs = env.prob_to_bool(probs)
        scores = env.get_scores(sln_xs)

        max_score, max_id = th.max(scores, dim=0)
        if max_score > best_score:
            best_score = max_score
            best_sln_x = sln_xs[max_id]
            print(f"best_score {best_score}  best_sln_x \n{env.bool_to_str(best_sln_x)}")


def check_sln_x():
    graph_name, sln_x = 'G14', SlnX14

    env = MCMCSim(graph_name=graph_name)
    best_sln_x = env.str_to_bool(sln_x)
    best_score = env.get_scores(best_sln_x.unsqueeze(0)).squeeze(0)
    print(f"NumNodes {env.num_nodes}  NumEdges {env.num_edges}")
    print(f"score {best_score}  sln_x \n{env.bool_to_str(best_sln_x)}")


def check_convert_sln_x():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 1
    graph_name = 'G14'
    # graph_name = 'g70'

    env = MCMCSim(graph_name=graph_name, gpu_id=gpu_id)

    x_prob = env.get_rand_probs(num_envs=num_envs)[0]
    x_bool = env.prob_to_bool(x_prob)

    x_str = env.bool_to_str(x_bool)
    print(x_str)
    x_bool = env.str_to_bool(x_str)

    assert all(x_bool == env.prob_to_bool(x_prob))


def exhaustion_search():
    import sys
    import time

    from tqdm import tqdm

    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 14
    th.manual_seed(0)

    th.set_grad_enabled(False)

    graph_names = ['syn_20_42', 'syn_20_47', 'syn_30_111', 'syn_30_129']
    for graph_name in graph_names:
        env = MCMCSim(graph_name=graph_name, gpu_id=gpu_id)
        dim = env.num_nodes

        best_score = -th.inf
        best_sln_x = None

        num_iter = 2 ** (dim - 1)
        print(f"NumNodes {env.num_nodes}  NumEdges {env.num_edges}  Total search num_iter {num_iter}")
        time.sleep(0.1)
        _num_envs = min(num_iter, num_envs)
        i_iter = tqdm(range(0, num_iter, _num_envs), ascii=True)
        all_score = np.empty(num_iter, dtype=np.int16)
        for i in i_iter:
            sln_xs = [env.int_to_bool(i + j) for j in range(_num_envs)]
            sln_xs = th.stack(sln_xs).to(env.device)
            scores = env.get_scores(sln_xs)

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_sln_x = sln_xs[max_id]
                i_iter.set_description(
                    f"best_score {best_score:6.0f}  best_sln_x {env.bool_to_str(best_sln_x)}")
                print()

            all_score[i:i + _num_envs] = scores.data.cpu().numpy()
        i_iter.close()

        num_best = np.count_nonzero(all_score == best_score.item())
        print(f"{graph_name}  NumNodes {env.num_nodes}  NumEdges {env.num_edges}  NumSearch {2 ** dim}  "
              f"best score {best_score:6.0f}  sln_x {env.bool_to_str(best_sln_x)}  count 2*{num_best}")

    """
    syn_20_42   NumNodes 20  NumEdges 42   NumSearch 1048576     best score     34  sln_x 0qBh   count 2*1
    syn_20_47   NumNodes 20  NumEdges 47   NumSearch 1048576     best score     35  sln_x 0UmA   count 2*24
    syn_30_111  NumNodes 30  NumEdges 111  NumSearch 1073741824  best score     82  sln_x FSgYW  count 2*4
    syn_30_129  NumNodes 30  NumEdges 129  NumSearch 1073741824  best score     92  sln_x BejfM  count 2*1
    """


"""find solution_x using optimizer"""


class OptimizerLSTM(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.rnn0 = nn.LSTM(inp_dim, mid_dim, num_layers=num_layers)
        self.mlp0 = nn.Linear(mid_dim, out_dim)
        self.rnn1 = nn.LSTM(1, 8, num_layers=num_layers)
        self.mlp1 = nn.Linear(8, 1)

    def forward(self, inp, hid0=None, hid1=None):
        tmp0, hid0 = self.rnn0(inp, hid0)
        out0 = self.mlp0(tmp0)

        d0, d1, d2 = inp.shape
        inp1 = inp.reshape(d0, d1 * d2, 1)
        tmp1, hid1 = self.rnn1(inp1, hid1)
        out1 = self.mlp1(tmp1).reshape(d0, d1, d2)

        out = out0 + out1
        return out, hid0, hid1


class UniqueBuffer:  # for GraphMaxCut
    def __init__(self, max_size: int, state_dim: int, gpu_id: int = 0):
        self.max_size = max_size
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.sln_xs = th.empty((max_size, state_dim), dtype=th.bool, device=self.device)
        self.scores = th.empty(max_size, dtype=th.int64, device=self.device)
        self.min_score = -th.inf
        self.min_index = None

    def update(self, sln_x, score):
        if (score < self.min_score) or th.any(th.all(sln_x.unsqueeze(0) == self.sln_xs, dim=1)):
            return False

        self.sln_xs[self.min_index] = sln_x
        self.scores[self.min_index] = score
        self.min_score, self.min_index = th.min(self.scores, dim=0)
        return True

    def save_or_load_history(self, cwd: str, if_save: bool):

        item_paths = (
            (self.sln_xs, f"{cwd}/buffer_sln_xs.npz"),
            (self.scores, f"{cwd}/buffer_scores.npz"),
        )

        if if_save:
            # print(f"| buffer.save_or_load_history(): Save {cwd}")
            for item, path in item_paths:
                np.savez_compressed(path, arr=item.data.cpu().numpy())

        elif all([os.path.isfile(path) for item, path in item_paths]):
            for item, path in item_paths:
                buf_item = np.load(path)['arr']
                print(f"| buffer.save_or_load_history(): Load {path}    {buf_item.shape}")

                max_size = buf_item.shape[0]
                max_size = min(self.max_size, max_size)
                item[:max_size] = th.tensor(buf_item[-max_size:], dtype=item.dtype, device=item.device)
            self.min_score, self.min_index = th.min(self.scores, dim=0)

    def init_with_random(self, env):
        probs = env.get_rand_probs(num_envs=self.max_size)
        sln_xs = env.prob_to_bool(probs)
        scores = env.get_scores(sln_xs)
        self.sln_xs[:] = sln_xs
        self.scores[:] = scores
        self.min_score, self.min_index = th.min(scores, dim=0)

    def print_sln_x_str(self, env):
        print(f"{'score':>8}  {'sln_x':8}")

        sort_ids = th.argsort(self.scores)
        for i in sort_ids:
            sln_x = self.sln_xs[i]
            score = self.scores[i]

            sln_x_str = env.bool_to_str(sln_x)
            enter_str = '\n' if len(sln_x_str) > 60 else ''
            print(f"score {score:8}  {enter_str}{sln_x_str}")

        ys = self.scores.sort().values.data.cpu().numpy()
        xs = th.arange(ys.shape[0]).data.cpu().numpy()
        plt.scatter(xs, ys)
        plt.title(f"max score {ys.max().item()}  top {ys.shape[0]}")
        plt.grid()
        plt.show()


def check_unique_list():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    graph_name, sln_x = 'G14', SlnX14

    env = MCMCSim(graph_name=graph_name, gpu_id=gpu_id)
    dim = env.num_nodes

    max_size = 4
    buffer = UniqueBuffer(max_size=max_size, state_dim=dim, gpu_id=gpu_id)
    buffer.init_with_random(env) if (buffer.min_score is None) else None

    sln_x = env.str_to_bool(SlnX14)
    score = env.get_scores(sln_x.unsqueeze(0)).squeeze(0)
    buffer.update(sln_x, score)
    print(buffer.scores)

    max_size = 2 ** 7
    buffer = UniqueBuffer(max_size=max_size, state_dim=dim, gpu_id=gpu_id)
    buffer.init_with_random(env) if (buffer.min_score is None) else None

    result_paths = [f"result_{graph_name}_{i}" for i in (5, 6, 7)]
    for result_path in result_paths:
        if os.path.exists(result_path):
            sln_xs = th.tensor(np.load(f"{result_path}/buffer_sln_xs.npz")['arr'], device=buffer.device)
            scores = th.tensor(np.load(f"{result_path}/buffer_scores.npz")['arr'], device=buffer.device)
            for i in range(sln_xs.shape[0]):
                buffer.update(sln_xs[i], scores[i])
        else:
            print(result_path)

    buffer.print_sln_x_str(env=env)
    # buffer.save_or_load_history(cwd=result_paths[0], if_save=True)


class Config:  # Demo
    def __init__(self, json_path: str = '', graph_name: str = 'gset_14', gpu_id: int = 0):
        # 数据超参数（路径，存储，预处理）
        self.json_path = './GraphMaxCut.json'  # 存放超参数的json文件。将会保存 class Config 里的所有变量 `vars(Config())`
        self.graph_name = graph_name
        self.gpu_id = gpu_id

        self.save_dir = f"./result_{graph_name}_{gpu_id}"

        '''GPU memory'''
        self.num_envs = 2 ** 6
        self.mid_dim = 2 ** 6
        self.num_layers = 2
        self.seq_len = 2 ** 6
        self.reset_gap = 2 ** 6

        '''exploit and explore'''
        self.learning_rate = 1e-3
        self.buf_size = 2 ** 8
        self.alpha_rate = 8
        self.diff_ratio = 0.1
        self.explore_weight = 1.0

        '''train and evaluate'''
        self.num_opti = 2 ** 16
        self.eval_gap = 2 ** 2

        if os.path.exists(json_path):
            self.load_from_json(json_path=json_path)
            self.save_as_json(json_path=json_path)
        vars_str = str(vars(self)).replace(", '", ", \n'")
        print(f"| Config\n{vars_str}")

    def load_from_json(self, json_path: str):
        with open(json_path, "r") as file:
            json_dict = json.load(file)
        for key, value in json_dict.items():
            setattr(self, key, value)

    def save_as_json(self, json_path: str):
        json_dict = vars(self)
        with open(json_path, "w") as file:
            json.dump(json_dict, file, indent=4)


def run_v1_find_sln_x_using_grad():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_name = 'G14'
    # graph_name = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2 ** 6

    '''init task'''
    env = MCMCSim(graph_name=graph_name, gpu_id=gpu_id)
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)

    '''loop'''
    best_score = -th.inf
    best_sln_x = probs
    start_time = time.time()
    for i in range(1, opt_num + 1):
        obj = env.get_objectives(probs).mean()
        obj.backward()

        grads = probs.grad.data
        probs.data.add_(-lr * grads).clip_(0, 1)

        if i % eval_gap == 0:
            sln_xs = env.prob_to_bool(probs)
            scores = env.get_scores(sln_xs)
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  {scores.max().item():9.0f}")

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_sln_x = sln_xs[max_id]
                print(f"best_score {best_score}  best_sln_x \n{env.bool_to_str(best_sln_x)}")

    print()
    print(f"best_score {best_score}  best_sln_x \n{env.bool_to_str(best_sln_x)}")


def run_v2_find_sln_x_using_adam():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 4
    graph_name = 'G14'
    # graph_name = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2 ** 6

    '''init task'''
    env = MCMCSim(graph_name=graph_name, gpu_id=gpu_id)
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)

    '''init opti'''
    opt_base = th.optim.Adam([probs, ], lr=lr)

    '''loop'''
    best_score = -th.inf
    best_sln_x = env.prob_to_bool(probs[0])
    start_time = time.time()
    for i in range(1, opt_num + 1):
        obj = env.get_objectives(probs).mean()
        opt_base.zero_grad()
        obj.backward()
        opt_base.step()

        probs.data.clip_(0, 1)

        if i % eval_gap == 0:
            sln_xs = env.prob_to_bool(probs)
            scores = env.get_scores(sln_xs)
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  {scores.max().item():9.0f}")

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_sln_x = sln_xs[max_id]
                print(f"best_score {best_score}  best_sln_x \n{env.bool_to_str(best_sln_x)}")

    print()
    print(f"best_score {best_score}  best_sln_x \n{env.bool_to_str(best_sln_x)}")


def run_v3_find_sln_x_using_opti():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_name = 'G14'
    # graph_name = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 8
    num_layers = 1
    seq_len = 2 ** 5
    reset_gap = 2 ** 6

    opt_num = int(2 ** 16 / num_envs)
    eval_gap = 2 ** 1

    '''init task'''
    env = MCMCSim(graph_name=graph_name, gpu_id=gpu_id)
    dim = env.num_nodes
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)
    obj = None
    hidden0 = None
    hidden1 = None

    '''init opti'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    opt_opti = OptimizerLSTM(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    '''loop'''
    best_score = -th.inf
    best_sln_x = env.prob_to_bool(probs[0])
    start_time = time.time()
    for i in range(1, opt_num + 1):
        if i % reset_gap == 0:
            probs = env.get_rand_probs(num_envs=num_envs)
            probs.requires_grad_(True)
            obj = None
            hidden0 = None
            hidden1 = None

        prob_ = probs.clone()
        updates = []

        for j in range(seq_len):
            obj = env.get_objectives(probs).mean()
            obj.backward()

            grad_s = probs.grad.data
            update, hidden0, hidden1 = opt_opti(grad_s.unsqueeze(0), hidden0, hidden1)
            update = (update.squeeze_(0) - grad_s) * lr
            updates.append(update)
            probs.data.add_(update).clip_(0, 1)
        hidden0 = [h.detach() for h in hidden0]
        hidden1 = [h.detach() for h in hidden1]

        updates = th.stack(updates, dim=0)
        prob_ = (prob_ + updates.mean(0)).clip(0, 1)
        obj_ = env.get_objectives(prob_).mean()

        opt_base.zero_grad()
        obj_.backward()
        opt_base.step()

        probs.data[:] = prob_

        if i % eval_gap == 0:
            sln_xs = env.prob_to_bool(probs)
            scores = env.get_scores(sln_xs)
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  "
                  f"max_score {scores.max().item():9.0f}  "
                  f"best_score {best_score}")

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_sln_x = sln_xs[max_id]

        if i % eval_gap * 256:
            print(f"best_score {best_score}  best_sln_x \n{env.bool_to_str(best_sln_x)}")

    print()
    print(f"best_score {best_score}  best_sln_x \n{env.bool_to_str(best_sln_x)}")


def run_v4_find_sln_x_using_opti_and_buffer():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    args = Config(json_path='', graph_name='gset_55', gpu_id=gpu_id)

    graph_name = args.graph_name
    save_dir = args.save_dir

    num_envs = args.num_envs
    mid_dim = args.mid_dim
    num_layers = args.num_layers
    seq_len = args.seq_len
    reset_gap = args.reset_gap

    lr = args.learning_rate
    buf_size = args.buf_size
    alpha_rate = args.alpha_rate
    diff_ratio = args.diff_ratio
    explore_weight = args.explore_weight

    num_opti = args.num_opti
    eval_gap = args.eval_gap

    '''init task'''
    env = MCMCSim(graph_name=graph_name, gpu_id=gpu_id)
    dim = env.num_nodes
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)
    sln_xs = env.prob_to_bool(probs)
    scores = env.get_scores(sln_xs)
    max_score, max_id = th.max(scores, dim=0)

    obj = None
    hidden0 = [th.zeros((num_layers, num_envs, mid_dim), dtype=th.float32, device=env.device) for _ in range(2)]
    hidden1 = [th.zeros((num_layers, num_envs * dim, 8), dtype=th.float32, device=env.device) for _ in range(2)]

    '''init opti'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    opt_opti = OptimizerLSTM(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr, amsgrad=False)
    # opt_base = th.optim.SGD(opt_opti.parameters(), lr=lr, momentum=0.99)

    '''inti buffer'''
    os.makedirs(save_dir, exist_ok=True)
    buffer = UniqueBuffer(max_size=buf_size, state_dim=dim, gpu_id=gpu_id)
    buffer.save_or_load_history(cwd=save_dir, if_save=False)
    buffer.init_with_random(env) if (buffer.min_score is None) else None

    '''init recorder'''
    best_score = -th.inf
    recorder = []

    '''loop'''
    start_time = time.time()
    pbar = tqdm(total=reset_gap * alpha_rate, ascii=True)
    mpl.use('Agg')  # Generating matplotlib graphs without a running X server [duplicate]
    for i in range(1, num_opti + 1):
        if i % reset_gap == 0:
            buffer.update(sln_x=sln_xs[max_id], score=max_score)
            probs = env.get_rand_probs(num_envs=num_envs)
            probs.requires_grad_(True)
            obj = None
            hidden0 = None
            hidden1 = None

        prob_ = probs.clone()
        updates = []

        for j in range(seq_len):
            obj = env.get_objectives(probs).mean()
            obj.backward()

            grad_s = probs.grad.data
            update, hidden0, hidden1 = opt_opti(grad_s.unsqueeze(0), hidden0, hidden1)
            update = (update.squeeze_(0) - grad_s) * lr
            updates.append(update)
            probs.data.add_(update).clip_(0, 1)
        hidden0 = [h.detach() for h in hidden0]
        hidden1 = [h.detach() for h in hidden1]

        updates = th.stack(updates, dim=0)
        prob_ = (prob_ + updates.mean(0)).clip(0, 1)
        obj_exploit = env.get_objectives(prob_).mean()

        if i % (reset_gap * alpha_rate) == 0:
            pbar = tqdm(total=reset_gap * alpha_rate, ascii=True)
        if i % (reset_gap * alpha_rate) > (reset_gap * (alpha_rate - 1)):
            buf_prob = buffer.sln_xs.float()
            obj_explore = (th.abs(prob_.unsqueeze(1) - buf_prob.unsqueeze(0)) - 0.5).clamp_min(0).sum(dim=(1, 2))
            obj_explore = obj_explore.clamp_max(dim / 2 * diff_ratio).mean() * explore_weight
        else:
            obj_explore = 0

        obj_ = obj_exploit + obj_explore
        opt_base.zero_grad()
        obj_.backward()
        opt_base.step()

        probs.data[:] = prob_

        sln_xs = env.prob_to_bool(probs)
        scores = env.get_scores(sln_xs)
        used_time = time.time() - start_time
        max_score, max_id = th.max(scores, dim=0)

        if max_score > best_score:
            best_score = max_score
            best_sln_x = sln_xs[max_id]
            best_sln_x_str = env.bool_to_str(best_sln_x)
            enter_str = '\n' if len(best_sln_x_str) > 60 else ''
            print(f"\nbest_score {best_score}  best_sln_x {enter_str}{best_sln_x_str}")

            recorder.append((i, best_score))
            recorder_ary = th.tensor(recorder)
            th.save(recorder_ary, f"{save_dir}/recorder.pth")
            buffer.save_or_load_history(cwd=save_dir, if_save=True)

            plt.plot(recorder_ary[:, 0], recorder_ary[:, 1])
            plt.scatter(recorder_ary[:, 0], recorder_ary[:, 1])
            plt.grid()
            plt.title(f"best_score {best_score}")
            plt.savefig(f"{save_dir}/recorder.jpg")
            plt.close('all')

        if i % eval_gap == 0:
            pbar.set_description(
                f"|{used_time:6.0f}  {i:6}  {obj.item():9.3f}  score {scores.max().item():6.0f}  {best_score}")
            pbar.update(eval_gap)
    pbar.close()


"""find solution_x using auto regression"""


class NetLSTM(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.rnn1 = nn.LSTM(inp_dim, mid_dim, num_layers=num_layers)
        self.mlp1 = nn.Sequential(nn.Linear(mid_dim, out_dim), nn.Sigmoid())

    def forward(self, inp, hid=None):
        tmp, hid = self.rnn1(inp, hid)
        out = self.mlp1(tmp)
        return out, hid


def sample_sln_x_using_auto_regression(num_envs, device, dim, opt_opti, if_train=True):
    hidden = None
    sample = th.zeros((num_envs, 1), dtype=th.float32, device=device)
    node_prob = th.zeros((num_envs, 1), dtype=th.float32, device=device)

    samples = []
    logprobs = []
    entropies = []

    samples.append(sample)
    for _ in range(dim - 1):
        obs = th.hstack((sample, node_prob))
        node_prob, hidden = opt_opti(obs, hidden)
        dist = Bernoulli(node_prob.squeeze(0))
        sample = dist.sample()

        samples.append(sample)
        if if_train:
            logprobs.append(dist.log_prob(sample))
            entropies.append(dist.entropy())

    samples = th.stack(samples).squeeze(2)
    sln_xs = samples.permute(1, 0).to(th.int)

    if if_train:
        logprobs = th.stack(logprobs).squeeze(2).sum(0)
        logprobs = logprobs - logprobs.mean()

        entropies = th.stack(entropies).squeeze(2).mean(0)
    return sln_xs, logprobs, entropies


def run_v1_find_sln_x_using_auto_regression():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 8
    # graph_name, num_limit = 'G14', sys.maxsize
    graph_name, num_limit = 'G14', 28

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 8
    num_layers = 1
    num_opt = int(2 ** 24 / num_envs)
    eval_gap = 2 ** 4
    print_gap = 2 ** 8

    alpha_period = 2 ** 10
    alpha_weight = 1.0

    '''init task'''
    env = MCMCSim(graph_name=graph_name, gpu_id=gpu_id)
    dim = env.num_nodes

    '''init opti'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    opt_opti = NetLSTM(inp_dim=2, mid_dim=mid_dim, out_dim=1, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    '''loop'''
    best_score = -th.inf
    best_sln_x = env.get_rand_probs(num_envs=1)[0]
    start_time = time.time()
    for i in range(num_opt):
        alpha = (math.cos(i * math.pi / alpha_period) + 1) / 2
        sln_xs, logprobs, entropies = sample_sln_x_using_auto_regression(num_envs, device, dim, opt_opti, if_train=True)
        scores = env.get_scores(probs=sln_xs).detach().to(th.float32)
        scores = (scores - scores.min()) / (scores.std() + 1e-4)

        obj_probs = logprobs.exp()
        obj = -((obj_probs / obj_probs.mean()) * scores + (alpha * alpha_weight) * entropies).mean()

        opt_base.zero_grad()
        obj.backward()
        opt_base.step()

        if i % eval_gap == 0:
            _sln_xs, _, _ = sample_sln_x_using_auto_regression(num_envs, device, dim, opt_opti, if_train=False)
            _scores = env.get_scores(_sln_xs)

            sln_xs = th.vstack((sln_xs, _sln_xs))
            scores = th.hstack((scores, _scores))

        max_score, max_id = th.max(scores, dim=0)
        if max_score > best_score:
            best_score = max_score
            best_sln_x = sln_xs[max_id]
            print(f"best_score {best_score}  best_sln_x \n{env.bool_to_str(best_sln_x)}")

        if i % print_gap == 0:
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  "
                  f"score {scores.max().item():6.0f}  {best_score:6.0f}  "
                  f"entropy {entropies.mean().item():6.3f}  alpha {alpha:5.3f}")

            if i % (print_gap * 256) == 0:
                print(f"best_score {best_score}  best_sln_x \n{env.bool_to_str(best_sln_x)}")


if __name__ == '__main__':
    # run_v1_find_sln_x_using_grad()
    # run_v2_find_sln_x_using_adam()
    # run_v3_find_sln_x_using_opti()
    run_v4_find_sln_x_using_opti_and_buffer()
    # run_v1_find_sln_x_using_auto_regression()
