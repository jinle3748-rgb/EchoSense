import os
from collections import defaultdict

LABELS = r'c:\Users\hp\Desktop\SoundCUDA-main\labels\dev'

file_speaker_count = {}
for filename in os.listdir(LABELS):
    if not filename.endswith('.rttm'):
        continue
    filepath = os.path.join(LABELS, filename)
    speakers = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                speakers.add(parts[7])
    file_speaker_count[filename] = len(speakers)

groups = defaultdict(list)
for fname, count in file_speaker_count.items():
    if count >= 6:
        groups['6人+'].append(fname)
    else:
        groups[f'{count}人'].append(fname)

print(f'RTTM 文件总数: {len(file_speaker_count)}')
print()
print(f'{"分组":<10} {"文件数":<10}')
print('-' * 20)
total = 0
for key in ['1人', '2人', '3人', '4人', '5人', '6人+']:
    cnt = len(groups.get(key, []))
    print(f'{key:<10} {cnt:<10}')
    total += cnt
print('-' * 20)
print(f'{"合计":<10} {total:<10}')
print()
for key in ['5人', '6人+']:
    files = groups.get(key, [])
    if files:
        print(f'{key} 文件 ({len(files)}个):')
        for f in files:
            fid = f.replace('.rttm', '')
            print(f'  {fid}  ({file_speaker_count[f]}人)')
        print()
