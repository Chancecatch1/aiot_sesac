n_list = ['최원칠','이서혁','aiot','새싹']
age_list = [30,29,34,72]
id_list = [125,2474,341,646]

# for문을 이용하여 3개의 리스트를 다음과 같은 리스트-딕셔너리 구조를 만들어라
key_list = ['name', 'age', 'id']
val_list = [n_list, age_list, id_list]
d_list = []

# name_dict =dict([(key, n) for key in key_list for n in n_list])
# print(name_dict)
# 3개의 딕셔너리를 만들자
# 각 딕셔너리안에는 key[i]:val[i][]
# { key_list[0]:n_list[0], key_list[1]:age_list[0], ... }
# { key_list[0]:n_list[1], key_list[1]:age_list[1], ... }
# { key_list[0]:n_list[2], key_list[1]:age_list[2], ... }
# { key_list[0]:n_list[3], key_list[1]:age_list[3], ... }

for i in range(len(n_list)): # 각 밸류 길이가 밸류 리스트 길이보다 짧기 때문
    # d_list.append( {key: val_list[j][i] for key in key_list} ) # 실패
    # d_list.append( {key : n_list[i] for key in key_list } ) # 이것도 실패
    # d_list.append( {key : age_list[i] for key in key_list } )
    # d_list.append( {key : id_list[i] for key in key_list } )
    dictionary = {}
    dictionary[key_list[0]] = n_list[i] # 이건 포문이랑 컴프리핸션으로 압축 못하나?
    dictionary[key_list[1]] = age_list[i]
    dictionary[key_list[2]] = id_list[i]
    d_list.append(dictionary)
    # print(dictionary)
print(d_list)

# 전체 age총합을 자유롭게 구해라
total_age = 0
for i in range(len(d_list)):
    total_age += d_list[i]['age']
print(total_age)

# for문과 if문을 이용하여 새싹의 나이를 35로 바꾸어라
for i in range(len(d_list)):
    if '새싹' in d_list[i]['name']:
        d_list[i]['age'] = 35
        print(d_list[i])
