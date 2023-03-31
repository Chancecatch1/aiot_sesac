# 비어있는 리스트 생성
a = []
print(a)
print('='*50)

# for문과 range, append 함수를 이용하여 a리스트를 다음과 같이 만들어라
# [10,11,12,13,14,15]
for i in range(10,16):
    a.append(i)
print(a)
print('='*50)

# extend함수를 이용하여 b리스트를 다음과 같이 만들어라
# [10, 11, 12, 13, 14, 15, 9, 8, 7]
b = a.copy()
b.extend([9,8,7])
print(b)
print('='*50)

# sort함수를 c리스트를 이용하여 다음과 같이 만들어라
# [7, 8, 9, 10, 11, 12, 13, 14, 15]
c = b.copy()
c.sort()
print(c)

# 중첩 리스트 다루기
# [3, 6, 13, 15, 18]
# [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15], [16,17,18,19,20] ]
# a 다음과 같은 두 개의 리스트를 for와 if문을 활용하여 다음과 같은 리스트로 만들어라(*은 문자열)
d = [3, 6, 13, 15, 18]
e = [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15], [16,17,18,19,20] ]

temp_list = []
final_list = []
for row in e:
    temp_list = []
    # temp_list.clear() 이 경우 마지막 temp_list 결과값으로만 저장되는 문제가 있ㅇㅁ
    # 참조의 문제, append는 데이터위치를 참조하는 것인데, 참조되는 temp_list의 값이 바뀌는 것이기 때문
    # =[]는 계속 메모리를 새로 할당하기 때문에 가능
    # print(id(temp_list))
    for col in row :
        if col in d :
            temp_list.append('*')
        else:
            temp_list.append(col)
    final_list.append(temp_list)
print(final_list)
print('='*50)

# enumerate를 활용해보자
data = e.copy()
for row in data :
    for index, col in enumerate(row):
        if col in d:
            row[index] = '*'
print(data)
