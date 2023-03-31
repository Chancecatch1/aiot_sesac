class Customer:

    def __init__(self, name, c_id):
        self.name = name
        self.c_id = c_id
        # 딕셔너리 구조로 계좌 여러개 저장
        self.accounts = {}

    # 계좌 딕셔너리에 계좌 추가 (0으로 초기화)
    def add_account(self, account, amount=0):
        if account in self.accounts:
            print(f'{account}계좌는 이미 존재합니다.')
        else :
            self.accounts = {account : amount}

    #해당 계좌에 금액 추가
    def add_amount(self, account, amount):
        if account in self.accounts:
            self.accounts[account] += amount
        else:
            print(f'입금 실패, {account}계좌가 존재하지 않습니다.')

    #해당 계좌에서 금액 제거
    def sub_amount(self, account, amount):
        if account in self.accounts:
            if self.accounts[account] >= amount:
                self.accounts[account] -= amount
            else:
                print(f'출금 실패, 계좌 금액이 부족합니다.')
        else:
            print(f'입금 실패, {account}계좌가 존재하지 않습니다.')

    # 전체 계좌금액 합계하는 함수 생성
    def get_total_amount(self):
        total = sum(self.accounts.values())
        return total

    def get_rat(self):
        if 100 < self.get_total_amount() <= 1000:
            rat = 'silver'
        elif 1000 < self.get_total_amount() <= 10000:
            rat =  'gold'
        elif 10000 < self.get_total_amount() <= 100000:
            rat =  'vip'
        elif 100000 < self.get_total_amount() :
            rat =  'vvip'
        else :
            rat =  'none'
        return rat

    def get_name(self):
        return self.name

    def get_id(self):
        return self.c_id

    # 전체 계좌 반환하는 함수 생성
    def get_all_accounts(self):
        return self.accounts

def create_customer():
    c_name = input('고객이름 입력:')
    c_id = input('고객번호 입력:')
    # 고객 인스턴스 생성
    customer = Customer(c_name, c_id)
    print('{} 고객이 생성 되었습니다.'.format(customer.get_name()))
    return customer

def show_list():
    for customer_obj in customer_list:
        print(f'고객이름: {customer_obj.get_name()} 고객번호: {customer_obj.get_id()}')

def search_customer(c_id):
    for customer_obj in customer_list:
        if c_id == customer_obj.get_id():
            customer = customer_obj
        else :
            customer = None
            print('고객 번호를 찾지 못했습니다.')
    return customer

def create_account():
    c_id = input('고객번호 입력:')
    # search_customer 사용하여 고객을 못찾으면 함수 끝냄
    customer = search_customer(c_id)
    if customer is None :
        breakpoint()
    else :
        account_num = input('계좌번호 입력:')
        customer.add_account(account_num)
        # 계좌 등록
        print('{} 고객의 {} 계좌가 등록되었습니다'
              .format(customer.get_name(),
                      account_num))

def show_customer():
    c_id = input('고객번호 입력:')
    # search_customer 사용하여 고객을 못찾으면 함수 끝냄
    customer = search_customer(c_id)
    if customer is None :
        breakpoint()
    else :
        print('{} 고객님 등급:{} 총금액:{} 계좌정보:{}'
              .format(customer.get_name(),
                      customer.get_rat(),
                      customer.get_total_amount(),
                      customer.get_all_accounts()
                      ))

def deposit():
    c_id = input('고객번호 입력:')
    # search_customer 사용하여 고객을 못찾으면 함수 끝냄
    customer = search_customer(c_id)
    if customer is None :
        breakpoint()
    else :
        account_num = input('계좌번호 입력:')
        amount = input('입금할 금액 입력:')
        # 계좌에 입금
        customer.add_amount(account_num, int(amount))

def withdraw():
    c_id = input('고객번호 입력:')
    # search_customer 사용하여 고객을 못찾으면 함수 끝냄
    customer = search_customer(c_id)
    if customer is None:
        breakpoint()
    else:
        account_num = input('계좌번호 입력:')
        amount = input('출금할 금액 입력:')
        customer.sub_amount(account_num, int(amount))


#####################   실행 ########################

if __name__ == '__main__':
    customer_list = []

    print('1 - 고객 생성')
    print('2 - 계좌 생성')
    print('3 - 입금')
    print('4 - 출금')
    print('5 - 생성된 고객 리스트 출력')
    print('6 - 고객 정보 출력')


    while True:
        print('--'*30)
        input_data = input('옵션을 입력해 주세요')
        if input_data == '1':
            customer = create_customer()
            #리스트에 고객 추가
            customer_list.append(customer)
        elif input_data == '2':
            create_account()
        elif input_data == '3':
            deposit()
        elif input_data == '4':
            withdraw()
        elif input_data == '5':
            show_list()
        elif input_data == '6':
            show_customer()
        else:
            print('존재하지 않는 명령어 입니다.')