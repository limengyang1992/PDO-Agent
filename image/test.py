from collections import Counter

def limit_duplicates(lst, limit):
    counter = Counter(lst)
    result = [item for item in lst if counter[item] <= limit or counter[item] > 0]
    return result

# 示例用法
my_list = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
limit = 3
result_list = limit_duplicates(my_list, limit)

print(result_list)
