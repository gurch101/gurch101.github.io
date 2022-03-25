---
title: Elements of Programming Interviews
date: 2021-10-10
description: Elements of Programming Interviews Summary
category: book summary
type: notes
---

### Arrays

```js
const a = [];
a.length // 0
a[100] // undefined
a[5] = 1
a.length // 6
a[0] // undefined

a.push(el); // add el to end of list
a.pop() // removes el from end of list

// splice changes the original array
const months = ['Jan', 'March', 'April', 'June'];
months.splice(1, 0, 'Feb');
// inserts at index 1
console.log(months);
// expected output: Array ["Jan", "Feb", "March", "April", "June"]

months.splice(4, 1, 'May');
// replaces 1 element at index 4
console.log(months);
// expected output: Array ["Jan", "Feb", "March", "April", "May"]

months.includes("March") // true
```

### Put Even Elements at Beginning of Array

```js
function evenFirst(arr) {
    let evenIdx = 0;
    let oddIdx = arr.length - 1;
    while(evenIdx < oddIdx) {
        if(arr[evenIdx] % 2 === 0) {
            evenIdx++;
        } else {
            let tmp = arr[evenIdx];
            arr[evenIdx] = arr[oddIdx];
            arr[oddIdx] = tmp;
            oddIdx--;
        }
    }
    return arr;
}
```


### Three-Way Partition (Dutch Flag Problem)

```js
function partition(arr, partitionIndex) {
    const val = arr[partitionIndex];
    let lo = 0;
    let equal = 0;
    let hi = arr.length;
    while(equal < hi) {
        if(arr[equal] < val) {
            let tmp = arr[lo];
            arr[lo] = arr[equal];
            arr[equal] = tmp;
            lo++;
        } else if(arr[equal] === val) {
            equal++;
        } else {
            hi--;
            let tmp = arr[hi];
            arr[hi] = arr[equal];
            arr[equal] = tmp;
        }
    }
    return arr;
}
```

### Add One To Integer Array

```js
function addOne(arr) {
    let carry = 0;
    arr[arr.length - 1]++;
    for(let i = arr.length - 1; i > 0; i--) {
        arr[i] += carry;
        carry = Math.floor(arr[i] / 10);
        arr[i] = arr[i] % 10;
        if(carry === 0) break;
    }

    arr[0] += carry;
    if(arr[0] === 10) {
        arr[0] = 1;
        arr.push(0);
    }
    return arr;
}
```

### Add Binary Strings

```js
function addBinary(b1, b2) {
    let carry = 0;
    let b1Arr = b1.split('').map(ch => parseInt(ch));
    let b2Arr = b2.split('').map(ch => parseInt(ch));
    for(let i = b1Arr.length - 1; i >= 0; i--) {
        b1Arr[i] += b2Arr[i] + carry;
        carry = Math.floor(b1Arr[i] / 2);
        b1Arr[i] = b1Arr[i] % 2;
    }
    if(carry != 0) {
        if(b1Arr[0] === 1) {
            // 11 + 11 = 110
            b1Arr[0] = 11;
        } else {
            // 1 + 1 = 10
            b1Arr[0] = 10;
        }
    }
    return b1Arr.join('');
}
```