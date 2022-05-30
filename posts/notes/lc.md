---
title: LeetCode
date: 2021-10-10
description: LeetCode Review
category: summary
type: notes
---

### Two Sum

Use a map to store the the index of each number, check if remainder exists.

### Sum Without Arithmetic Operators

& tells you bitwise carry, ^ tells you bitwise sum, set y to carry << 1, continue until y = 0

### Best time to buy/sell stock

keep track of minprice, update max price to be max(profit, curPrice - minprice)

### Contains duplicate

keep numbers in a set, check if val already exists in set

### Hamming Weight

num & 1, shift n >> 1, repeat 32 times

### Product of all numbers except self

maintain two arrays of running product from left and running product from right, then answer is product of left[i - 1] * right.reverse()[i + 1]

alt:

```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        // get length of array
        int n = nums.length; 
        // Initiailze ans to store the result
        int[] ans = new int[n]; 
        // we know that the product of any element in nums = 
        // product[left elements] * product[right elements]
        // we need to compute the right product of each index of nums
        // and just use ans to store this temporary result. 
        // Base Case : Since right produce of the right most element is 1. 
        ans[n-1] = 1; 
        // for each remaining elements compute right product.
        for ( int i = n -2 ; i >= 0 ; i--){
            ans[i] = nums[i+1] * ans[i+1]; 
        }
        // left product can be tracked using a single pointer and 
        // updating it with every iteration.
        int lp = 1; 
        for ( int i = 0 ;i < n ; i++){
            // replace right product in ans with both left and right products. 
            ans[i] = lp * ans[i]; 
            lp = lp * nums[i]; 
        }
        return ans; 
    }
}
```

### Counting Bits
number of bits in ABCD = number of bits in ABC + number of bits in D
```
arr[i] = arr[i >>> 1] + (i & 1)
```

### Climb Stairs

recursion + memoization

```js
var cache = [0, 1, 2]
var climbStairs = function(n) {

    if(cache[n] !== undefined) return cache[n];
    cache[n] = climbStairs(n - 1) + climbStairs(n - 2);
    return cache[n];
};
```

### Clone Graph

DFS

```js
var cloneGraph = function(node) {
    if(!node) return node;
    const nodeClones = {[node.val]: new Node(node.val)};
    const toVisit = [node];
    while(toVisit.length > 0) {
        const curr = toVisit.pop();

        for(let i = 0; i < curr.neighbors.length; i++) {
            if(nodeClones[curr.neighbors[i].val] === undefined) {
                toVisit.push(curr.neighbors[i]);
            }
            nodeClones[curr.neighbors[i].val] = nodeClones[curr.neighbors[i].val] || new Node(curr.neighbors[i].val);
            nodeClones[curr.val].neighbors.push(nodeClones[curr.neighbors[i].val]);
        }
    }
    
    return nodeClones[node.val];
};
```

### Reverse a Linked List

Maintain pointers to tail, oldHead, newHead. Draw a diagram

```js
curr = a
newHead = curr;
oldHead = curr;
while(curr && curr.next) {
    newHead = curr.next;
    curr.next = newHead.next;
    newHead.next = oldHead;
    oldHead = newHead;
}

return newHead;
```

### Set Matrix Zeroes

use first row/col as indicator whether row/col should be made zero. Keep track of whether the first col/row should be made zero. Create helper functions to make row/col by index zero.

### Max SubArray

```js
var maxSubArray = function(nums) {
    maxSum = nums[0];
    curSum = maxSum;
    i = 1;
    while(i < nums.length) {
        curSum = Math.max(nums[i], curSum + nums[i]);
        maxSum = Math.max(maxSum, curSum);
        i++;
    }
    return maxSum;
};
```

### Max Depth of Binary Tree

recursion - max(maxDepth of left, maxDepth of right) + 1

### Coin Change

given a bag of coins and an amount, find the minimum number of coins needed to make amount where dupe coins are allowed.

dp problem where you keep track of min number of coins for each amount 0 to amount

```js
var coinChange = function(coins, amount) {
    const dp = [0];
    for(let i = 1; i < amount +  1; i++) {
        dp.push(Number.MAX_VALUE);
    }
    
    for(let i = 1; i < amount + 1; i++) {
        for(let j = 0; j < coins.length; j++) {
            if(coins[j] <= i) {
                dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
            }
        }
    }
    if(dp[amount] === Number.MAX_VALUE) return -1;
    return dp[amount];
};
```

### Insert Interval

```java
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        ArrayList<int[]> ans=new ArrayList<>();
        
        //Add all pairs less than newInterval
        int i=0;
        for(i=0;i<intervals.length;i++){
            if(intervals[i][0]<newInterval[0]){
                ans.add(intervals[i]);
            }
            else{
                break;
            }
        }
        
        
        // I have come here that means i have found the position to add newInterval
        
        //CASE 1 --> When newInterval is the first element to get added in ans or it is not merging with any element
        if(ans.size()==0 || (newInterval[0]>ans.get(ans.size()-1)[1])){
            ans.add(newInterval);
        }
        else{
            //merging
            int[] lastinterval=ans.get(ans.size()-1);
            lastinterval[1]=Math.max(lastinterval[1],newInterval[1]);
        }
        
        // CASE 2--> Now since newIterval has been added but the remaining intervals are left so lets add them up in ans
        
        while(i<intervals.length){
              int[] lastInterval=ans.get(ans.size()-1);
              if(lastInterval[1]>=intervals[i][0]){
                  //merging
               lastInterval[1]=Math.max(lastInterval[1],intervals[i][1]);
              }
              else{
                      ans.add(intervals[i]);
              }
            i++;
        }
        return ans.toArray(new int[ans.size()][]);
    }
}
```
### Linked List Cycle

use fast and slow pointer. Proof - if fast is one behind, it will equal slow on next iteration; fast can't be one ahead because they would've been on the same node on the previous iteration.

### Spiral Matrix

maintain count of num visited - loop until all visited (matrix.length * matrix[0].length), maintain flags for direction and min/max indices for range

### Length of longest non-repeating substring

keep track of start and index of most recent occurence of char

```js
var lengthOfLongestSubstring = function(s) {
    var start = 0;
    var cur = 0;
    var maxLength = 0;
    var length = 0;
    var indices = {};
    while(cur < s.length) {
        // if we haven't encountered the char, increment length
        // or if we encountered the char outside the start of the window, increment length
        if(indices[s[cur]] === undefined || indices[s[cur]] < start) {
            length++;
        } else {
            maxLength = Math.max(maxLength, length);
            // new start is 1 + last occurence of char
            start = indices[s[cur]] + 1;
            // +1 since we have zero based indices
            length = cur - start + 1;
        }
        indices[s[cur]] = cur;
        cur++;
    }
    return Math.max(maxLength, length);
}
```

### Missing Number

summation formula OR iterate from 1 - n: total += i - this is still O(n).

### Max Product

iterate in both directions, keep track of max, reset running product whenever zero is encountered

```js
var maxProduct = function(nums) {

    let max = nums[0];
    if(nums.length === 1) return max;
    
    let product = 1;
    for(let i = 0; i < nums.length; i++) {
        product *= nums[i];
        max = Math.max(max, product);
        if(nums[i] === 0) product = 1;
    }
    
    product = 1;
    for(let i = nums.length - 1; i >= 0; i--) {
        product *= nums[i];
        max = Math.max(max, product);
        if(nums[i] === 0) product = 1;
    }
    return max;
};
```

### Find Minimum in Rotated Sorted Array

use binary search, check if on pivot point, return val, else check if pivot is in the left or the right.

```js
var findMin = function(nums) {
    let lo = 0;
    let hi = nums.length - 1;
    let mid = Math.floor(nums.length / 2);
    let minVal = nums[mid];
    while(lo < hi) {
        mid = lo + Math.floor((hi - lo) / 2);
        minVal = Math.min(minVal, nums[mid]);
        if(mid + 1 < nums.length && nums[mid + 1] <= minVal) {
            return nums[mid + 1];
        } else {
            if(nums[mid] > nums[hi]) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    }
    
    return minVal;
};
```

### Search in Rotated Sorted Array

```js
 */
var isRotationOnLeft = function(nums, lo, mid, hi) {
    return nums[mid] < nums[lo];
}
var isRotationOnRight = function(nums, lo, mid, hi) {
    return nums[mid] > nums[hi];
}

var search = function(nums, target) {
    let lo = 0;
    let hi = nums.length - 1;
    let mid = Math.floor((hi - lo) / 2)
    if(nums.length === 1) return nums[0] === target? 0 : -1;
    while(lo <= hi) {
        mid = lo + Math.floor((hi - lo) / 2);
        if(nums[mid] === target) return mid;
        if(isRotationOnLeft(nums, lo, mid, hi)) {
            if(target <= nums[hi] && target > nums[mid]) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        } else if(isRotationOnRight(nums, lo, mid, hi)) {
            if(target >= nums[lo] && target < nums[mid]) {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        } else if(target > nums[mid]) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
};
```

### 3Sum

traps: javascript sort casts each element to string before sorting!

```js
var threeSum = function(nums) {
    var sums = {};
    var triplets = [];
    var uniques = {};
    nums.sort((a, b) => a - b);
    for(let i = 0; i < nums.length; i++) {
        const firstNum = nums[i];
        let start = i + 1;
        let end = nums.length - 1;
        while(start < end && start < nums.length) {
            const sum = firstNum + nums[start] + nums[end];
            if(sum === 0) {
                if(uniques[firstNum + " " + nums[start] + " " + nums[end]] === undefined) {
                    triplets.push([firstNum, nums[start], nums[end]]);
                    uniques[firstNum + " " + nums[start] + " " + nums[end]] = true;
                }
                start++;
                end--;
            } else if(sum < 0) {
                start++;
            } else {
                end--;
            }
        }
    }
    return triplets;
};
```

### Container with most water

two pointers in an array

```js
var maxArea = function(height) {
    let curMax = 0;
    let lo = 0;
    let hi = height.length - 1;
    while(lo < hi) {
        const curHeight = Math.min(height[lo], height[hi]);
        const curLength = hi - lo;
        curMax = Math.max(curMax, curHeight * curLength);
        // keep the tallest side fixed
        // if we kept the shortest side fixed, we would have a shorter height and shorter width which is guaranteed less that curMax
        if(height[lo] > height[hi]) {
            hi--;
        } else {
            lo++;
        }
    }
    
    return curMax;
};
```

### Reverse Bits

num >>> 0 gets twos complement of a negative number

```js
var reverseBits = function(n) {
    let rev = 0;
    let bit = 0;
    while(bit < 32) {
        if(n & (1 << bit)) {
            rev |= (1 << (31 - bit));
        }
        bit++;
    }
    return rev >>> 0;
};
```

### Course Schedule

```js
var canFinish = function(numCourses, prerequisites) {
    let g = {};
    let inboundCts = {};
    let toVisit = [];
    // build graph
    for(let i = 0; i < prerequisites.length; i++) {
        const [to, from] = prerequisites[i];
        g[from] = g[from] || [];
        g[from].push(to);
        g[to] = g[to] || [];
        inboundCts[to] = inboundCts[to] || 0;
        inboundCts[to]++;
    }
    // initial nodes to visit are those with no pre-reqs
    const nodes = Object.keys(g);
    for(let i = 0; i < nodes.length; i++) {
        if(inboundCts[nodes[i]] === undefined) {
            toVisit.push(nodes[i]);
        }
    }
    let numVisited = 0;
    while(toVisit.length > 0) {
        const n = toVisit.pop();
        numVisited++;
        for(let i = 0; i < g[n].length; i++) {
            const neighbor = g[n][i];
            inboundCts[neighbor]--;
            if(inboundCts[neighbor] === 0) {
                toVisit.push(neighbor);
            }
        }
    }
    // count those without any pre-reqs as visited
    for(let i = 0; i < numCourses; i++) {
        if(g[i] === undefined) {
            numVisited++;
        }
    }
    return numVisited === numCourses;
};
```