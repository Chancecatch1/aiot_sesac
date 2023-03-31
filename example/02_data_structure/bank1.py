class Customer:

    def __init__(self, name, id):
       self.name = name
       self.id = id
       self.amount = 0

    def add_amount(self, add):
        self.add = add
        self.amount += add

    def sub_amount(self, sub):
        self.sub = sub
        if self.amount < self.sub:
            print('잔액이 부족합니다.')
        else :
            self.amount -= sub

    def get_name(self):
       return self.name

    def get_id(self):
       return self.id

    def get_amount(self):
       return self.amount

    def get_rat(self):
        if 100 < self.amount <= 1000:
            return 'silver'
        elif 1000 < self.amount <= 10000:
            return 'gold'
        elif 10000 < self.amount <= 100000:
            return 'vip'
        elif 100000 < self.amount :
            return 'vvip'
        else :
            return 'none'

if __name__ == '__main__': # 메인함수의 선언, 시작을 의미

    c_name = input('고객 이름 입력:')
    c_id = input('고객 번호 입력:')
    customer = Customer(c_name, c_id)
    print(customer.get_name() + ' 고객이 생성 되었습니다.') # '{고객이름} 고객이 생성 되었습니다' 출력

    amount = int(input('입금 금액 입력'))
    customer.add_amount(amount)

    amount = int(input('출금 금액 입력'))
    customer.sub_amount(amount)
    customer.get_rat()
    print('='*50)
    print(f"고객 이름 {customer.get_name()}, 고객 번호 {customer.get_id()}, 고객 등급 {customer.get_rat()}, 계좌 금액 {customer.get_amount()}") # '고객이름, 고객아이디, 계좌금액' 출력
    print('고객 생성 완료')
    print('='*50)
    count = 0
    for i in range(50):
        count += 1
        if count % 3 == 0:
            print(f'===={count}회 정기 점검====')
        else :
            print(f'===={count}회 반복 작업====')
            num = input('입금은 1번, 출금은 2번, 조회는 3번, 종료는 4번을 눌러주세요')
            if num == '1':
                amount = int(input('입금 금액을 입력하세요.'))
                customer.add_amount(amount)
            elif num == '2':
                amount = int(input('출금 금액을 입력하세요.'))
                customer.sub_amount(amount)
            elif num == '3':
                print(
                    f"고객 이름 {customer.get_name()}, 고객 번호 {customer.get_id()}, 고객 등급 {customer.get_rat()}, 계좌 금액 {customer.get_amount()}")  # '고객이름, 고객아이디, 계좌금액' 출력
            elif num == '4':
                print('종료합니다')
                break
            else:
                print('잘못된 입력입니다')

