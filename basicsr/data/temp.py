'''
@Author  ：zww
@Date    ：2025/6/4 10:03 
@Description : *
'''

from typing import List


def groupAnagrams(strs: List[str]) -> List[List[str]]:
    result = {}
    for item in strs:
        key = "".join(sorted(item))
        if key in result:
            result[key].append(item)
        else:
            result[key] = [item]
    return list(result.values())

strs = ["eat","tea","tan","ate","nat","bat"]
result = groupAnagrams(strs)
print(result)
