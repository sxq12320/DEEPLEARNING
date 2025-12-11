# 给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。 
# 
#  如果数组中不存在目标值 target，返回 [-1, -1]。 
# 
#  你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [5,7,7,8,8,10], target = 8
# 输出：[3,4] 
# 
#  示例 2： 
# 
#  
# 输入：nums = [5,7,7,8,8,10], target = 6
# 输出：[-1,-1] 
# 
#  示例 3： 
# 
#  
# 输入：nums = [], target = 0
# 输出：[-1,-1] 
# 
#  
# 
#  提示： 
# 
#  
#  0 <= nums.length <= 10⁵ 
#  -10⁹ <= nums[i] <= 10⁹ 
#  nums 是一个非递减数组 
#  -10⁹ <= target <= 10⁹ 
#  
# 
#  Related Topics 数组 二分查找 👍 3078 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        ans = []
        left = 0
        right = n-1

        while left <= right:
            mid = (left + right)//2
            middle = nums[mid]
            if middle >= target: # right 的右边一定的大于等于target
                right = mid - 1
            elif middle < target: # left的左边一定小于target
                left = mid + 1
        ans.append(left)
        if left == n or nums[left] != target:
            return [-1 , -1]

        left = 0
        right = n-1
        while left <= right:
            mid = (left + right) // 2
            middle = nums[mid]
            if middle > target: # right的右边一定大于target
                right = mid -1
            elif middle <= target: # left的左边一定小于等于target
                left = mid +1
        ans.append(right)
        return ans

# leetcode submit region end(Prohibit modification and deletion)
