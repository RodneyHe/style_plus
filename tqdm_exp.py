from tqdm import tqdm
from time import sleep

# update the progress bar
# for i in range(10):
#     pbar = tqdm(range(100), ncols=80, desc=f"epoch {i}:")
#     for j in pbar:
#         sleep(0.1)

# for i in tqdm(range(100), ncols=80, bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt}   {elapsed}/{remaining}"):
#     sleep(0.1)

# 自定义一个进度条样式函数
def custom_bar(current, total, width=40):
    progress = int(width * current / total)
    return '[' + '=' * progress + '>' + '.' * (width - progress - 1) + ']'

# 使用自定义的进度条样式函数
bar_format = '{l_bar}{bar} | {n_fmt}/{total_fmt} | {elapsed}/{remaining}'
with tqdm(total=100, bar_format=bar_format, bar_format='{l_bar}' + custom_bar(0, 100)) as pbar:
    for i in range(100):
        pbar.update(1)
        pbar.set_description(f"Processing {i}")
        pbar.set_postfix(status='OK')
        sleep(0.1)