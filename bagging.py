a = [1, 0, 1, 0, 0, 1, 1, 0, 0 ,0]
val = [0.81, 0.61, 0.41, 0.21, 0.81, 0.61, 0.41, 0.21,0.81, 0.61]

thres = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

arr1 = [0] * len(a)
arr2 = [0] * len(a)
arr3 = [0] * len(a)
arr4 = [0] * len(a)
arr5 = [0] * len(a)
arr6 = [0] * len(a)

def gen_arr(val, thr, arr):
    for i in range(len(val)):
        if(val[i] >= thres[thr]):
            arr[i] = 1
        else:
            arr[i] = 0
    return 0


gen_arr(val, 0, arr1)          
gen_arr(val, 1, arr2)          
gen_arr(val, 2, arr3)          
gen_arr(val, 3, arr4)          
gen_arr(val, 4, arr5)          
gen_arr(val, 5, arr6)  

tp = [0] * len(a)
fn = [0] * len(a)
tn = [0] * len(a)
fp = [0] * len(a)

def check_val(arr, j):
    for i in range(len(a)):
        if a[i] == 1 and arr[i] == 1:
            tp[j] = tp[j] + 1
        elif a[i] == 1 and arr[i] == 0:
            fn[j] = fn[j] + 1
        elif a[i] == 0 and arr[i] == 1:
            fp[j] = fp[j] + 1
        elif a[i] == 0 and arr[i] == 0:
            tn[j] = tn[j] + 1
    return 0

check_val(arr1, 0)
check_val(arr2, 1)
check_val(arr3, 2)
check_val(arr4, 3)
check_val(arr5, 4)
check_val(arr6, 5)

def bagging(res):
    ct1 = 0
    ct0 = 0
    for i in range(len(res)):
        if res[i] == 1:
            ct1 = ct1 + 1
        elif res[i] == 0:
            ct0 = ct0 + 1
    
    if ct1 < ct0:
        print("According to bagging vote mechanism, it's reported to be NEGATIVE CLASSIFICATION in " ,res)
    else:
        print("According to bagging vote mechanism, it's reported to be POSITIVE CLASSIFICATION in ", res)

    return 0

bagging(tp)
bagging(tn)
bagging(fp)
bagging(fn)



