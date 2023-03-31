import copy
# users: 유저의 정보가 유저id를 키값으로 하여 저장되어있는 딕셔너리
users = {123:['최원칠','관악구'],
         124:['이서혁','구로구'],
         125:['새싹','용산구'],
         126:['aiot','용산구']}
# products: 상품의 이름이 상품의 id값을 키값으로 하여 저장되어있는 딕셔너리
# products의 구조가 상품 키에 이름과 재고개수가 묶인 딕셔너리가 매핑된 형태로 수정되어있다.
products = {'a':{'name':'노트북', 'count':10},
            'b':{'name':'자전거', 'count':10},
            'c':{'name':'샴푸', 'count':10},
            'd':{'name':'셔츠', 'count':0},
            'e':{'name':'초코렛', 'count':10}}

# orders: 주유저아이디와 구매상품들이 딕셔너리로 묶여있는 주문 내역이 순서대로 저장되어있는 리스트
orders = [{'user_id':123, 'products':['a','c']},
          {'user_id':125, 'products':['e']},
          {'user_id':124, 'products':['b','d','e']},
          {'user_id':126, 'products':['a']}]
delivery = {}

# 위 데이터들을 활용하여 다음과 같은 배달 정보 리스트를 만드시오
# {'관악구': [('최원칠', '노트북'), ('최원칠', '샴푸')],
#  '용산구': [('새싹', '초코렛'), ('aiot', '노트북')],
# '구로구': [('이서혁', '자전거'), ('이서혁', '초코렛')]}

# orders에서 유저 아이디와 프로덕트 값을 추출해서 개별 리스트를 만든다
products_list = []
name_list = []
locate_list = []

for i in range(len(orders)):
    products_list.append(orders[i]['products'])
    name_list.append(users[orders[i]['user_id']][0])
    locate_list.append(users[orders[i]['user_id']][1])
print(name_list)
print(products_list)
print(locate_list)
print('='*50)

# 프로덕트 이름 가져오기
products_name = copy.deepcopy(products_list)
for i in range(len(products_name)):
    for j in range(len(products_name[i])):
        products_name[i][j] = products[products_name[i][j]]['name']
print(products_list)
print(products_name)
print('='*50)

# 지역구 : [(이름,상품), (이름, 상품)]  합쳐서 딕셔너리로 변환
# 포인트 : 지역구 기준,  , 주문 순서 주의(유저아이디 섞여있음), 튜플 컨프리핸션
for i in range(len(locate_list)):
    if locate_list[i] not in delivery.keys(): # 딕셔너리에 없는 키에 대한 밸류 입력
        delivery[locate_list[i]] = [(name_list[i], products_name[i][j]) for j in range(len(products_name[i]))
                                    # 재고 체크 조건, 카운팅 기능 없음
                                    if products_name[i][j] == products[products_list[i][j]]['name']
                                        and products[products_list[i][j]]['count'] > 0
                                    ]

    else: # 딕셔너리에 있는 키에 대한 밸류 추가(딕셔너리변환시 동일한 키가 중복될 경우 set되기 때문에, 누락된 밸류를 다시 추가해준다)
        delivery[locate_list[i]] += [(name_list[i], products_name[i][j]) for j in range(len(products_name[i]))
                                     # 재고 체크 조건, 카운팅 기능 없음
                                     if products_name[i][j] == products[products_list[i][j]]['name']
                                        and products[products_list[i][j]]['count'] > 0
                                     ]

print(delivery)
print('='*50)