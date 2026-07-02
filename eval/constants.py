import argparse


class Dataset(argparse.Namespace):
    def __init__(self, name, ids, pattern, shape, padding, padding_pos):
        self.name = name
        self.ids = ids
        self.pattern = pattern
        self.shape = shape
        self.padding = padding
        self.padding_pos = padding_pos
        self.samples = dict()
        self.paths = set()


lesav_ids = [
    12, 15, 33, 34, 37, 42, 49, 53, 90, 111, 119, 224,
    240, 275, 279, 304, 318,
    367, 370, 486, 529, 586
]
lesav_dataset = Dataset(
    pattern='{id_}.png',
    ids=[str(i) for i in lesav_ids],
    name='LES-AV',
    shape=(1444, 1620),
    padding=((6, 7), (0, 0), (0, 0)),
    padding_pos=1,
)

lestest_ids = [
    224,
    240, 275, 279, 304, 318,
    367, 370, 486, 529, 586
]
lestest_dataset = Dataset(
    pattern='{id_}.png',
    ids=[str(i) for i in lestest_ids],
    name='LES-test',
    shape=(1444, 1620),
    padding=((6, 7), (0, 0), (0, 0)),
    padding_pos=1,
)

rite_test_dataset = Dataset(
    pattern='{id_}.png',
    ids=[f"{i:02d}" for i in range(1, 21)],
    name='RITE-test',
    shape=(584, 565),
    padding=((4, 4), (5, 6), (0, 0)),
    padding_pos=2,
)


hrf_ids = [
    '01_dr', '01_g', '01_h', '02_dr', '02_g', '02_h', '03_dr', '03_g', '03_h',
    '04_dr', '04_g', '04_h', '05_dr', '05_g', '05_h',
]
hrf_dataset = Dataset(
    pattern='{id_}.png',
    ids=hrf_ids,
    name='HRF',
    shape=(2336, 3504),
    padding=None,
    padding_pos=None,
)


lf_private_ids = [
    'OS_20241117125448.999', 'OS_20241117152443.233', 'OS_20241128154515.988',
    'OS_20241128154931.846', 'OS_20241207152728.716', 'OS_20241207153109.020',
    'OS_20241207154335.715', 'OS_20241228153741.529', 'OS_20241228160852.858',
    'OS_20241228161230.375', 'OS_20241228162159.555', 'OS_20241229150048.798',
    'OS_20241229150457.808', 'OS_20241229150942.298', 'OS_20241229151558.932',
    'OS_20241229163705.122', 'OS_20241229164805.829', 'OS_20250101120635.678',
    'OS_20250111114127.463', 'OS_20250111114600.431', 'OS_20250112103327.015',
    'OS_20250112103626.064',
]
lf_private_dataset = Dataset(
    pattern='{id_}.png',
    ids=lf_private_ids,
    name='LF_private',
    shape=None,
    padding=None,
    padding_pos=None,
)


dataset_factory = {
    'RITE-test': rite_test_dataset,
    'LESAV': lesav_dataset,
    'HRF-test': hrf_dataset,
    'LES-test' : lestest_dataset,
    'LF_private': lf_private_dataset,
}
