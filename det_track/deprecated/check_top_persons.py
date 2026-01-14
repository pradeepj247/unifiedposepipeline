import numpy as np

data = np.load('test_data/canonical_persons.npz', allow_pickle=True)
persons = data['persons']
sorted_persons = sorted(persons, key=lambda p: len(p['frame_numbers']), reverse=True)

print('Top 12 persons by frame count:')
print(f"{'Person ID':<12} {'Frames':<10} {'First Frame':<12} {'Last Frame':<12}")
print('-'*50)
for p in sorted_persons[:12]:
    print(f"{int(p['person_id']):<12} {len(p['frame_numbers']):<10} {p['frame_numbers'][0]:<12} {p['frame_numbers'][-1]:<12}")
