import torch

arr = torch.tensor([3, 1, 2])
sorted_indices = torch.argsort(arr)

print("Original tensor:", arr)
print("Sorted indices:", sorted_indices)
print("Sorted tensor:", arr[sorted_indices])