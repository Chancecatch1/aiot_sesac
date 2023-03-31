#  강사님 정답
users = {123:['최원칠','관악구'],
         124:['이서혁','구로구'],
         125:['새싹','용산구']}

products = {'a':'노트북',
            'b':'자전거',
            'c':'샴푸',
            'd':'셔츠',
            'e':'초코렛'}

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

for order in orders:
    uid = order['user_id']
    u_name = users[uid][0]
    u_locate = users[uid][1]
    p_ids = order['products']
    for p in p_ids:
        d = {}
        d['name'] = u_name
        d['product'] = products[p]
        d['locate'] = u_locate
        delivery.append(d)

print(delivery)