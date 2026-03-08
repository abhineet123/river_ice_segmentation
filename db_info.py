def irange(a, b):
    return list(range(a, b + 1))


class RiverIceValInfo:
    class DBSplits:
        def __init__(self):
            self.all = irange(0, 0)

    sequences = {
        0: ('images', 564),
    }


class RiverIceInfo:
    class DBSplits:
        def __init__(self):
            self.all = irange(0, 0)

    sequences = {
        0: ('images', 50),
    }


class COCOInfo:
    class DBSplits:
        def __init__(self):
            self.all = irange(0, 1)
            self.train = irange(0, 0)
            self.val = irange(1, 1)

    sequences = {
        0: ('train2017', 118287),
        1: ('val2017', 5000),
    }


class CityscapesInfo:
    class DBSplits:
        def __init__(self):
            self.all = irange(0, 26)
            self.test = irange(0, 5)
            self.train = irange(6, 23)
            self.val = irange(24, 26)

    sequences = {
        # test
        0: ('test/berlin', 544),
        1: ('test/bielefeld', 181),
        2: ('test/bonn', 46),
        3: ('test/leverkusen', 58),
        4: ('test/mainz', 298),
        5: ('test/munich', 398),
        # train
        6: ('train/aachen', 174),
        7: ('train/bochum', 96),
        8: ('train/bremen', 316),
        9: ('train/cologne', 154),
        10: ('train/darmstadt', 85),
        11: ('train/dusseldorf', 221),
        12: ('train/erfurt', 109),
        13: ('train/hamburg', 248),
        14: ('train/hanover', 196),
        15: ('train/jena', 119),
        16: ('train/krefeld', 99),
        17: ('train/monchengladbach', 94),
        18: ('train/strasbourg', 365),
        19: ('train/stuttgart', 196),
        20: ('train/tubingen', 144),
        21: ('train/ulm', 95),
        22: ('train/weimar', 142),
        23: ('train/zurich', 122),
        # val
        24: ('val/frankfurt', 267),
        25: ('val/lindau', 59),
        26: ('val/munster', 174),
    }


class IPSCInfo:
    class DBSplits:
        def __init__(self):
            self.all = irange(0, 30)

    sequences = {
        0: ('all_frames_roi_7777_10249_10111_13349', 127),
        1: ('all_frames_roi_8094_13016_11228_15282', 127),
        2: ('all_frames_roi_9861_9849_12861_11516', 127),
        3: ('all_frames_roi_10127_9782_12527_11782', 127),
        4: ('all_frames_roi_10161_9883_13561_12050', 127),
        5: ('all_frames_roi_11927_12517_15394_15550', 127),
        6: ('all_frames_roi_12094_17082_16427_20915', 127),
        7: ('all_frames_roi_12527_11015_14493_12615', 127),
        8: ('all_frames_roi_12794_8282_14661_10116', 127),
        9: ('all_frames_roi_12994_10915_15494_12548', 127),
        10: ('all_frames_roi_16627_11116_18727_12582', 127),
        11: ('roi_4961_15682_7127_16949', 127),
        12: ('roi_6661_13749_9061_14816', 127),
        13: ('roi_7594_11916_9927_13149', 127),
        14: ('roi_7694_8682_10194_9682', 127),
        15: ('roi_7727_10749_9961_11749', 127),
        16: ('roi_8461_17782_10194_19016', 127),
        17: ('roi_9261_13449_11494_14382', 127),
        18: ('roi_10228_10182_12394_11915', 127),
        19: ('roi_10494_8849_12494_9849', 127),
        20: ('roi_11661_13082_13594_14849', 127),
        21: ('roi_12394_17282_14327_20782', 127),
        22: ('roi_12761_10682_14894_11782', 127),
        23: ('roi_12861_8815_15027_10115', 127),
        24: ('roi_12961_11916_14661_12816', 127),
        25: ('roi_13894_13749_16527_15316', 127),
        26: ('roi_14094_17682_15894_19749', 127),
        27: ('roi_15827_11316_17627_12749', 127),
        28: ('roi_15927_17249_17627_19582', 127),
        29: ('roi_17094_13782_19127_16348', 127),
        30: ('roi_17861_11316_19661_12616', 127),
    }


class IPSCDevInfo:
    class DBSplits:
        def __init__(self):
            self.all = irange(0, 19)
            self.g1 = irange(0, 2)
            self.g2 = irange(3, 5)
            self.g3 = irange(6, 13)
            self.g4 = irange(14, 19)
            self.g4s = irange(14, 18)
            self.test = [20, ]
            self.nd03 = [21, ]

            self.g2_4 = self.g2 + self.g3 + self.g4
            self.g3_4 = self.g3 + self.g4
            self.g3_4s = self.g3 + self.g4s

    sequences = {
        # g1
        0: ('Frame_101_150_roi_7777_10249_10111_13349', 6),
        1: ('Frame_101_150_roi_12660_17981_16026_20081', 3),
        2: ('Frame_101_150_roi_14527_18416_16361_19582', 3),
        # g2
        3: ('Frame_150_200_roi_7644_10549_9778_13216', 46),
        4: ('Frame_150_200_roi_9861_9849_12861_11516', 47),
        5: ('Frame_150_200_roi_12994_10915_15494_12548', 37),
        # g3
        6: ('Frame_201_250_roi_7711_10716_9778_13082', 50),
        7: ('Frame_201_250_roi_8094_13016_11228_15282', 50),
        8: ('Frame_201_250_roi_10127_9782_12527_11782', 50),
        9: ('Frame_201_250_roi_11927_12517_15394_15550', 48),
        10: ('Frame_201_250_roi_12461_17182_15894_20449_1', 30),
        11: ('Frame_201_250_roi_12527_11015_14493_12615', 50),
        12: ('Frame_201_250_roi_12794_8282_14661_10116', 50),
        13: ('Frame_201_250_roi_16493_11083_18493_12549', 49),
        # g4
        14: ('Frame_251__roi_7578_10616_9878_13149', 25),
        15: ('Frame_251__roi_10161_9883_13561_12050', 24),
        16: ('Frame_251__roi_12094_17082_16427_20915', 25),
        17: ('Frame_251__roi_12161_12649_15695_15449', 25),
        18: ('Frame_251__roi_12827_8249_14594_9816', 25),
        19: ('Frame_251__roi_16627_11116_18727_12582', 25),
        # test
        20: ('Test_211208', 59),
        21: ('nd03', 414),
    }


class IPSCPatchesInfo:
    class DBSplits:
        def __init__(self):
            self.all = irange(0, 19)
            self.g1 = irange(0, 2)
            self.g2 = irange(3, 5)
            self.g3 = irange(6, 13)
            self.g4 = irange(14, 19)
            self.g4s = irange(14, 18)
            self.test = [20, ]
            self.nd03 = [21, ]

            self.g2_4 = self.g2 + self.g3 + self.g4
            self.g3_4 = self.g3 + self.g4

    sequences = {
        # g1
        0: ('Frame_101_150_roi_7777_10249_10111_13349', 250),
        1: ('Frame_101_150_roi_12660_17981_16026_20081', 352),
        2: ('Frame_101_150_roi_14527_18416_16361_19582', 192),
        # g2
        3: ('Frame_150_200_roi_7644_10549_9778_13216', 269),
        4: ('Frame_150_200_roi_9861_9849_12861_11516', 446),
        5: ('Frame_150_200_roi_12994_10915_15494_12548', 233),
        # g3
        6: ('Frame_201_250_roi_7711_10716_9778_13082', 351),
        7: ('Frame_201_250_roi_8094_13016_11228_15282', 350),
        8: ('Frame_201_250_roi_10127_9782_12527_11782', 274),
        9: ('Frame_201_250_roi_11927_12517_15394_15550', 725),
        10: ('Frame_201_250_roi_12461_17182_15894_20449_1', 345),
        11: ('Frame_201_250_roi_12527_11015_14493_12615', 293),
        12: ('Frame_201_250_roi_12794_8282_14661_10116', 357),
        13: ('Frame_201_250_roi_16493_11083_18493_12549', 257),
        # g4
        14: ('Frame_251__roi_7578_10616_9878_13149', 125),
        15: ('Frame_251__roi_10161_9883_13561_12050', 143),
        16: ('Frame_251__roi_12094_17082_16427_20915', 150),
        17: ('Frame_251__roi_12161_12649_15695_15449', 225),
        18: ('Frame_251__roi_12827_8249_14594_9816', 100),
        19: ('Frame_251__roi_16627_11116_18727_12582', 125),
        # test
        20: ('Test_211208', 59),
        # test
        21: ('nd03', 414),

    }


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
