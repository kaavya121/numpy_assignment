#Q1 How to Creating a 3x3 Identity Matrix with Float Data Type? 
import numpy as np 
identity_matrix = np.identity(3,dtype = float)
print(identity_matrix) 

#Q2 Create a 1D Array with Random Values between 0 and 1 
import numpy as np 
arr = np.random.rand(5)
print(arr)

#Q3 Create a 2D Array with Random Integer Values. 
import numpy as np 
arr_1 = np.random.randint(0,10,size = (2,2))
print(arr_1)

#Q4 Creating an Array Using a Custom Function. 
import numpy as np
def custom_func(x, y):
    return x + y

array_shape = (3, 3)  
arr_2 = np.fromfunction(custom_func, array_shape, dtype=int)

print(arr_2)

#Q5 Reshaping a 1D Array into a 2D Array  
import numpy as np 
arr_3 = np.array([1,2,3,4,5,6])
arr_4 = arr_3.reshape(2,3)
print(arr_4)

#Q6.How to Creating a 3x3 Array of Ones? 
import numpy as np
arr_5 = np.ones((3,3))
print(arr_5)

'''
Q7. How to get the common items between two pythons
NumPy? Input:
a = np. array ([1,2,3,2,3,4,3,4,5,6])
b = np. array ([7,2,10,2,7,4,9,4,9,8])
Expected Output:
array ([2, 4]) 
'''
import numpy as np
a = np. array ([1,2,3,2,3,4,3,4,5,6])
b = np. array ([7,2,10,2,7,4,9,4,9,8])
c = np.intersect1d(a,b)
print(c)

'''
Q8. From array a remove all items present in
array b Input:
a = np. array ([1,2,3,4,5])
b = np. array ([5,6,7,8,9])
Expected Output:
array ([1,2,3,4]
'''
import numpy as np 
a = np. array ([1,2,3,4,5])
b = np. array ([5,6,7,8,9])
c = np.setdiff1d(a,b)
print(c)

'''
9. Limit the number of items printed in python NumPy array a to a maximum of
6 elements.
a = np. arrange (15)
Expected Output:
array ([ 0, 1, 2, ..., 12, 13, 14] 
'''

import numpy as np
a = np.arange(15)
b = a[:6]
print(b)


'''
10. Drop all nan values from a 1D NumPy
array Input:
np. array ([1,2,3, np.nan,5,6,7, np.nan])
Desired Output:
array ([ 1., 2., 3., 5., 6., 7.]) 
'''
import numpy as np 
a = np. array ([1,2,3, np.nan,5,6,7, np.nan])
b = a[~np.isnan(a)]
print(b)

#11. Question: Create a 1D NumPy array of the first 20 natural numbers and a2D NumPy array of shape (4, 5) with values ranging from 1 to 20 
import numpy as np
a = np.arange(1,21)
print(a)
b = a.reshape(4,5)
print(b)

'''
12. : Given a 3D NumPy array of shape (2, 3, 4), find its shape, size,
number of dimensions, and data type. Change its data type to float64 and verify
the change
'''
import numpy as np 
a = np.random.randint(1, 10, size=(2, 3, 4))
s = a.shape
si = a.size
dim = a.ndim
data = a.dtype
print(s)
print(si)
print(dim)
print(data)
b = a.astype(np.float64)
print(b)

'''
Q13 Reshape a 1D array of 12 elements into a 3x4 2D array and then
flatten it back into a 1D array using ravel (). Verify that the flattened array
matches the original. 
'''
import numpy as np 
a = np.arange(1,13)
print(a)
b = a.reshape(3,4)
print(b)
c = b.ravel()
print(c)

'''
Q14 Given two arrays, a = np. array ([1, 2, 3]) and b = np. array ([4,
5, 6]), perform element-wise addition, subtraction, multiplication, and
division. Explain the behavior when dividing by zero. 
'''
a = np. array ([1, 2, 3])
b = np. array ([4, 5, 6])
c = np.add(a,b)
print(c)
d = np.subtract(a,b)
print(d)
e = np.multiply(a,b)
print(e)
f = np.divide(a,b)
print(f)


'''
Q15 Create a 2D array of shape (3, 1) and a 1D array of length 3. Perform
element-wise addition using broadcasting. Explain how broadcasting rules apply
in this scenario.
''' 
a = np.array([[1],[2],[3]])
print(a)
b = np.arange(4,7)
print(b)
c = np.add(a,b)
print(c)
#allow the smaller array to be broadcast across the larger one, ensuring that they have compatible shapes for these operations.

'''
Q16  Generate a random 2D array of integers between 0 and 10. Use conditional operators to create a Boolean mask identifying elements greater than
5. Replace all elements greater than 5 with the value 5. 
'''
a = np.random.randint(0,10, size=(4,5))
print(a)
mask = a > 5
print("Boolean Mask (True where elements > 5):",mask)
a[mask] = 5
print("Array After Replacing Elements > 5 with 5:",a[mask])


'''
Given a 4x4 array of random integers, use indexing and slicing
to extract:
o The entire second row
o The last column
o The subarray consisting of the first two rows and first two columns
'''
a = np.random.randint(0,17, size=(4,4))
print(a)
print("1. ",a[1,:])
print("2. ",a[:,3])
print("3. ",a[0:2,0:2])


'''
Q18 Describe a practical example where NumPy can be used in EDA,
AI, ML, and DL, and implement a NumPy solution for a simple task in each
area. 
'''
#doubt

'''
Q19 Create a random 4x4 matrix and use NumPy to compute its
eigenvalues and eigenvectors. Verify the eigenvalues by reconstructing the
matrix. 
'''
a = np.random.randint(0,17,size=(4,4))
eigenvalues, eigenvectors = np.linalg.eig(a)
print(eigenvalues, eigenvectors)

'''
Q20 Create a 1D array of 27 elements and reshape it into a 3x3x3
3D array. Flatten it back into a 1D array and compare the flattened array
with the original. 
'''
a = np.random.randint(0,11,size=27)
print(a)
b = a.reshape(3,3,3)
print(b)

'''
Q21 Perform matrix multiplication of two 2D arrays using np.dot ()
and @. Compare the results and performance of both methods using a large
dataset. 
'''
a = np.array([1,2,3],[5,6,7])
b = np.array([3,4,5],[7,8,9])
c = np.dot(a,b)
print(c)


'''
Q22 Create a 3D array of shape (2, 1, 4) and a 2D array of shape (4,
1). Perform an element-wise operation using broadcasting and explain the
result. Use np. new axis to achieve the same result without broadcasting. 
'''
a = np.random.randint(0,11,size=(3,3,3))
print(a)
vector = np.array([10, 20, 30])
mat = a + vector 
print(mat)

'''
Q23 : Generate a 2D array of random floats between 0 and 1. Use
conditional operators to create a Boolean mask for values less than 0.5.
Replace these values with their squares and leave the rest unchanged. 
'''
a = np.random.random((2,3))
print(a)
mask = a > 0.5
print("Boolean Mask (True where elements > 5):",mask)
a[mask] = 0.5
print("Array After Replacing Elements > 5 with 5:",a[mask])

'''
Q24 : Given a 5x5 array of sequential integers, use slicing to:
o Extract the diagonal elements
o Replace the elements of the middle row with zeros
o Flip the array vertically and horizontally 
'''
matrix = np.array([[ 1,  2,  3,  4,  5],
                   [ 6,  7,  8,  9, 10],
                   [11, 12, 13, 14, 15],
                   [16, 17, 18, 19, 20],
                   [21, 22, 23, 24, 25]])
print(matrix)
c= np.diag(b)
print("1.",c)
middle = matrix.shape[0] // 5
matrix[middle, :] = 0
print(matrix)



'''
Q25 Create a 4D array of shape (2, 3, 4, 5) with random integers.
Use advanced slicing to extract a subarray and compute the mean along a
specified axis. 
'''
a = np.random.randint(0,15,size=(2,3,4,5))
print(a)

'''
Given an array of shape (10, 20), reshape it to (20, 10) and (5,
40). Discuss the impact on the array's shape, size, and dimensionality. 
'''
a = np.random.randint(0,15,size=(10,20))
print(a)
b = a.reshape(20,10)
print(b)
c= a.reshape(5,40)
print(c)

'''
Generate a large 2D array and demonstrate the use of np. reshape
() and unravel () to manipulate its shape for various linear algebra operations. 
'''
rows, cols = 6, 8  
large_array = np.arange(rows * cols).reshape(rows, cols)

print("Original 2D Array:")
print(large_array)

new_shape = (4, 12) 
reshaped_array = np.reshape(large_array, new_shape)

print("\nReshaped Array (4x12):")
print(reshaped_array)
flattened_array = np.ravel(large_array)
print("\nFlattened Array (1D):")
print(flattened_array)


'''
Given a 6x6 matrix, use advanced indexing and slicing to extract
the upper triangular part of the matrix and set the lower triangular part to zero.
Verify the result. 
'''
#doubt