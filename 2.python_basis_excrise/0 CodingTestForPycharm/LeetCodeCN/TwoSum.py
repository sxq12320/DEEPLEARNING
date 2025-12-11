
#IMPORTANT!! Submit Code Region Begin(Do not remove this line)
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        sum = 0
        for i in range(0 ,n,1):
            for j in range(i+1 ,n ,1):
                sum = nums[i] + nums[j]
                if sum == target:
                    return [i , j]
        
#IMPORTANT!! Submit Code Region End(Do not remove this line)