import glob

import torch
from tqdm import tqdm


class ATDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dict):
        self.dataset_dict = dataset_dict

    def __len__(self):
        return len(self.dataset_dict)

    def __getitem__(self, idx):
        return self.dataset_dict[idx]


class ATFileDataset(torch.utils.data.Dataset):
    """Mimic ATDataset but keep files separate."""

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.files = glob.glob(os.path.join(dataset_dir, "*.pkl"))
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return load_pkl(self.files[idx])


def collate_fn(batch):
    if len(batch) == 1:
        d = {k: batch[0][k][0].float() for k in batch[0].keys() if k != "name"}
        d["name"] = [batch[0]["name"]]
        d["aatype"] = batch[0]["aatype"].long()
        return d
    else:
        # Needs work for padding to work
        max_len = max([b["s"][0].shape[-2] for b in batch])
        d = {}
        for prot in batch:
            for k, v in prot.items():
                if k not in d:
                    d[k] = []
                v = v[0]
                if k == "s_initial":
                    len_diff = max_len - v.shape[-2]
                    new_value = torch.cat(
                        [v, torch.zeros(v.shape[0], len_diff, v.shape[-1]).float()],
                        dim=-2,
                    )
                    d[k].append(new_value)
                elif k != "name":
                    try:
                        len_diff = max_len - v.shape[-2]
                        new_value = torch.cat(
                            [
                                v,
                                torch.zeros(
                                    v.shape[0], v.shape[1], len_diff, v.shape[-1]
                                ).float(),
                            ],
                            dim=-2,
                        )
                        d[k].append(new_value)
                    except Exception as e:
                        print(e)
                        print(k)
                        print(v.shape)
                        print(max_len)
                        print(len_diff)
                        raise e
                else:
                    d[k].append(prot[k][0])

        d = {k: torch.stack(d[k]).float() for k in d.keys() if k != "name"}

        d["name"] = [b["name"] for b in batch]
        return d


def load_pkl(fn):
    with open(fn, "rb") as f:
        _d = pickle.load(f)
    return _d


def load_multiple_pickles(pattern):
    files = glob.glob(pattern)
    files.sort()
    all_data = (load_pkl(fn) for fn in files)
    updated_data = {}
    cur_idx = 0
    print("Loading data...")
    for fn, d in tqdm(zip(files, all_data), total=len(files)):
        updated_data.update({cur_idx: d})
    return updated_data
