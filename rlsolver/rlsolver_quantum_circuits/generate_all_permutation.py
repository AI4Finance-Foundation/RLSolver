for a in range(0,5):
    for b in range(0,5):
        for c in range(0,5):
            for d in range(0, 5):
                for e in range(0, 5):
                    if a != b and a != c and a != d and a != e and b != c and b != d and b != e and c != d and c != e and d != e:
                        print('[',a,',',b,',',c,',',d,',',e,"]", end=', ')
