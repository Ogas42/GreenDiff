def get_tqdm():
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(iterable=None, **kwargs):
            return iterable
    return tqdm

