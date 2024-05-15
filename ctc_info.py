def irange(a, b):
    return list(range(a, b + 1))


class CTCInfo:
    class DBSplits:
        def __init__(self):
            self.all_r = irange(0, 19)
            self.bf_r = irange(0, 3)
            self.bf1_r = irange(0, 1)
            self.bf2_r = irange(2, 3)
            self.dic_r = irange(4, 5)
            self.fluo_r = irange(6, 15)
            self.fluo1_r = irange(6, 11)
            self.fluo2_r = irange(12, 15)
            self.huh_r = irange(6, 7)
            self.gow_r = irange(8, 9)
            self.sim_r = irange(10, 11)
            self.hela_r = irange(14, 15)
            self.phc_r = irange(16, 19)
            self.phc1_r = irange(16, 17)
            self.phc2_r = irange(18, 19)

            self.all_e = irange(20, 39)
            self.bf_e = irange(20, 23)
            self.bf1_e = irange(20, 21)
            self.bf2_e = irange(22, 23)
            self.dic_e = irange(24, 25)
            self.fluo_e = irange(26, 35)
            self.fluo1_e = irange(26, 31)
            self.fluo2_e = irange(32, 35)
            self.huh_e = irange(26, 27)
            self.gow_e = irange(28, 29)
            self.sim_e = irange(30, 31)
            self.hela_e = irange(34, 35)
            self.phc_e = irange(36, 39)
            self.phc1_e = irange(36, 37)
            self.phc2_e = irange(38, 39)

            self.all = self.all_r + self.all_e
            self.bf = self.bf_r + self.bf_e
            self.bf1 = self.bf1_r + self.bf1_e
            self.bf2 = self.bf2_r + self.bf2_e
            self.dic = self.dic_r + self.dic_e
            self.fluo = self.fluo_r + self.fluo_e
            self.fluo1 = self.fluo1_r + self.fluo1_e
            self.fluo2 = self.fluo2_r + self.fluo2_e
            self.huh = self.huh_r + self.huh_e
            self.gow = self.gow_r + self.gow_e
            self.sim = self.sim_r + self.sim_e
            self.hela = self.hela_r + self.hela_e
            self.phc = self.phc_r + self.phc_e
            self.phc1 = self.phc1_r + self.phc1_e
            self.phc2 = self.phc2_r + self.phc2_e

    sequences = {
        # train
        0: ('BF-C2DL-HSC_01', 1764),
        1: ('BF-C2DL-HSC_02', 1764),
        2: ('BF-C2DL-MuSC_01', 1376),
        3: ('BF-C2DL-MuSC_02', 1376),

        4: ('DIC-C2DH-HeLa_01', 84),
        5: ('DIC-C2DH-HeLa_02', 84),

        6: ('Fluo-C2DL-Huh7_01', 30),
        7: ('Fluo-C2DL-Huh7_02', 30),

        8: ('Fluo-N2DH-GOWT1_01', 92),
        9: ('Fluo-N2DH-GOWT1_02', 92),

        10: ('Fluo-N2DH-SIM_01', 65),
        11: ('Fluo-N2DH-SIM_02', 150),

        12: ('Fluo-C2DL-MSC_01', 48),
        13: ('Fluo-C2DL-MSC_02', 48),

        14: ('Fluo-N2DL-HeLa_01', 92),
        15: ('Fluo-N2DL-HeLa_02', 92),

        16: ('PhC-C2DH-U373_01', 115),
        17: ('PhC-C2DH-U373_02', 115),
        18: ('PhC-C2DL-PSC_01', 300),
        19: ('PhC-C2DL-PSC_02', 300),

        # test

        20: ('BF-C2DL-HSC_Test_01', 1764),
        21: ('BF-C2DL-HSC_Test_02', 1764),
        22: ('BF-C2DL-MuSC_Test_01', 1376),
        23: ('BF-C2DL-MuSC_Test_02', 1376),

        24: ('DIC-C2DH-HeLa_Test_01', 115),
        25: ('DIC-C2DH-HeLa_Test_02', 115),

        26: ('Fluo-C2DL-Huh7_Test_01', 30),
        27: ('Fluo-C2DL-Huh7_Test_02', 30),

        28: ('Fluo-N2DH-GOWT1_Test_01', 92),
        29: ('Fluo-N2DH-GOWT1_Test_02', 92),

        30: ('Fluo-N2DH-SIM_Test_01', 110),
        31: ('Fluo-N2DH-SIM_Test_02', 138),

        32: ('Fluo-C2DL-MSC_Test_01', 48),
        33: ('Fluo-C2DL-MSC_Test_02', 48),

        34: ('Fluo-N2DL-HeLa_Test_01', 92),
        35: ('Fluo-N2DL-HeLa_Test_02', 92),

        36: ('PhC-C2DH-U373_Test_01', 115),
        37: ('PhC-C2DH-U373_Test_02', 115),
        38: ('PhC-C2DL-PSC_Test_02', 300),
        39: ('PhC-C2DL-PSC_Test_01', 300),
    }
