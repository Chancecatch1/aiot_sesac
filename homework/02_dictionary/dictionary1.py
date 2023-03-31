# users: 유저의 정보가 유저id를 키값으로 하여 저장되어있는 딕셔너리
users = {123:['최원칠','관악구'],
         124:['이서혁','구로구'],
         125:['새싹','용산구']}
# products: 상품의 이름이 상품의 id값을 키값으로 하여 저장되어있는 딕셔너리
products = {'a':'노트북',
            'b':'자전거',
            'c':'샴푸',
            'd':'셔츠',
            'e':'초코렛'}
# orders: 주유저아이디와 구매상품들이 딕셔너리로 묶여있는 주문 내역이 순서대로 저장되어있는 리스트
orders = [{'user_id':123, 'products':['a','c']},
         {'user_id':125, 'products':['e']},
         {'user_id':124, 'products':['b','d','e']}]
delivery = []
# 위 데이터들을 활용하여 다음과 같은 배달 정보 리스트를 만드시오
# [ {'name': '최원칠', 'product': '노트북', 'locate': '관악구'},
# {'name': '최원칠', 'product': '샴푸', 'locate': '관악구'},
# {'name': '새싹', 'product': '초코렛', 'locate': '용산구'},
# {'name': '이서혁', 'product': '자전거', 'locate': '구로구'},
# {'name': '이서혁', 'product': '셔츠', 'locate': '구로구'},
# {'name': '이서혁', 'product': '초코렛', 'locate': '구로구'} ]

# orders에서 유저 아이디와 프로덕트 값을 추출해서 개별 리스트를 만든다?
products_list = []
name_list = []
locate_list = []
key_list = ['name', 'product', 'locate']
for i in range(len(orders)):
    products_list.append(orders[i]['products'])
    name_list.append(users[orders[i]['user_id']][0])
    locate_list.append(users[orders[i]['user_id']][1])
print(name_list)
print(products_list)
print(locate_list)
print('='*50)

# 프로덕트 매핑??
for i in range(len(products_list)):
    for j in range(len(products_list[i])):
        # print(products_list[i][j])
        products_list[i][j] = products[products_list[i][j]]
        # print(products_list[i][j])
print(products_list)
print('='*50)

# 키랑 합쳐서 딕셔너리로 변환, 포인트는 '상품배송정보'이므로 '상품'단위로 리스트업
for i in range(len(products_list)):
    for j in range(len(products_list[i])):
        dictionary = {}
        dictionary[key_list[0]] = name_list[i]
        dictionary[key_list[1]] = products_list[i][j]
        dictionary[key_list[2]] = locate_list[i]
        delivery.append(dictionary)
print(delivery)
print('='*50)
