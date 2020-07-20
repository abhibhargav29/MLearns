def threeSum(self, nums):
    res = []        
    n = len(nums)   
    nums.sort()     
        
    for i in range(n-2):
        #if the array has only +ve nums
        if nums[i] > 0:
            break
            
        #if aray is sorted, below condition would imply that this i will lead to duplicates
        if i > 0 and nums[i] == nums[i-1]:
            continue
            
        #Two pointers to check for sum
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l += 1
            elif s > 0:
                r -= 1
            else:
                res.append([nums[i], nums[l], nums[r]])
                    
                while l < r and nums[l] == nums[l+1]:
                    l += 1
                while l < r and nums[r] == nums[r-1]:
                    r -= 1
                l += 1
                r -= 1
    return res
