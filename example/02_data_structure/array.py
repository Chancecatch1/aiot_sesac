import numpy as np
# a. [1,2,'a']의 리스트를 배열로 바꿔서 타입과 함께 출력 하여라
mat3 = [1,2,'a']
mat3 = np.asarray(mat3)

print(mat3, type(mat3))

# b. (5,10) 모양을 가지는 랜덤한 배열을 생성하고 0번축의 0~2, 1번축의 4~5 인덱스에
# 해당하는 부분 배열을 가져와 모양을 출력하여라
# rand1 = np.random.rand(5, 10)
# print(rand1)
rand2 = np.random.randint(0, 10, size=(5,10))
print(rand2)
sub_rand = rand2[0, 0:3]
print(sub_rand, sub_rand .shape)
sub_rand = rand2[0, 4:6]
print(sub_rand, sub_rand.shape)

# C. 다음 구조의 매트릭스를 배열로 바꾼후 1번축의 가장 큰값들을 추출하여라 (np max함수)
matrix = [
     [1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]
]
print(matrix, type(matrix))
amat = np.asarray(matrix)
print(amat, type(amat))
amat_max1 = np.max(amat, axis=1) # input 데이터로 배열을 넣어야함
print(amat_max1, amat_max1.shape)
mat_max1 = np.max(matrix, axis=1) # 리스트와 같은 이터러블도 가능 자동으로 바꿔주는 듯
print(mat_max1, mat_max1.shape)

# D. c번의 배열에서 0번축의 가장 큰값들을 추출하여라 (np max함수)

mat_max0 = np.max(matrix, axis=1)
print(mat_max0, mat_max0.shape)

# amat = np.asarray(matrix)

# E. (5,6)의 모양을 가지는 랜덤한 배열을 생성하고 b의 (5,10)모양의 배열과 1번축을 합쳐주어라,
# 최종 모양은 (5,16)이 되어야 한다.
rand3 = np.random.randint(0, 10, size=(5,6))
print(rand3)
cat_rand = np.concatenate((rand2, rand3), axis=1)
print(cat_rand, cat_rand.shape)