# PowerShell — creates the file
@"
def fibonacci(n):
    """
    Returns the nth Fibonacci number using recursion.
    Base cases: fibonacci(0) = 0, fibonacci(1) = 1
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


def binary_search(arr, target):
    """
    Binary search on a sorted array.
    Returns index of target or -1 if not found.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"@ | Out-File -FilePath documents\sample.py -Encoding utf8