# LeetCode

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
