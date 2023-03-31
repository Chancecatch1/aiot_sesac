class Customer:

    def __init__(self, name, c_id):
        self.name = name
        self.c_id = c_id
        # 딕셔너리 구조로 계좌 여러개 저장
        self.accounts = {}

    #해당 계좌에 금액 추가
    def add_amount(self, a_id, amount):
        if a_id in self.accounts:
            self.accounts[a_id] += amount
        else:
            print('해당 계좌가 없습니다')
    #해당 계좌에서 금액 제거
    def sub_amount(self, a_id, amount):
        if a_id in self.accounts:
            if self.accounts[a_id] > amount:
                self.accounts[a_id] -= amount
            else:
                print('계좌의 금액이 부족합니다.')
        else:
            print('해당 계좌가 없습니다.')
    #계좌 딕셔너리에 계좌 추가 (0으로 초기화)
    def add_account(self, a_id):
        if a_id in self.accounts:
            print('해당 계좌가 이미 존재합니다')
        else:
            self.accounts[a_id] = 0

    # 전체 계좌금액 합계
    def get_total_amount1(self,):
        total_amount = 0
        for a_id in self.accounts:
            total_amount += self.accounts[a_id]

        return total_amount

    def get_total_amount2(self,):
        amounts = []
        for a_id in self.accounts:
            amounts.append(self.accounts[a_id])

        return sum(amounts)

    def get_rat(self):
        total_amount = self.get_total_amount1()
        if total_amount > 100000:
            rat = 'vvip'
        elif total_amount > 10000:
            rat = 'vip'
        elif total_amount > 1000:
            rat = 'gold'
        elif total_amount > 100:
            rat = 'silver'
        else:
            rat = 'bronze'

        return rat

    def get_name(self):
        return self.name

    def get_id(self):
        return self.c_id

    def get_all_accounts(self):
        return self.accounts


def create_customer():
    c_name = input('고객 이름 입력:')
    c_id = input('고객 번호 입력:')
    customer = Customer(c_name, c_id)
    print('{} 고객이 생성 되었습니다.'.format(customer.get_name()))
    return customer


def show_list():
    for c in customer_list:
        print('고객이름:{} 고객번호:{}'
              .format(c.get_name(),
                      c.get_id(),
                      ))

def search_customer(c_id):
    for customer in customer_list:
        if customer.get_id() == c_id:
            print('고객이름:{} 고객번호:{}'
                  .format(customer.get_name(),
                          customer.get_id()
                          ))
            return customer

    print('고객 번호를 찾지 못했습니다.')
    return None

def create_acount():
    print('고객 번호 입력:')
    c_id = input()
    customer = search_customer(c_id)
    if customer is None:
        return

    print('계좌번호 입력:')
    acount_num = input()
    customer.add_account(acount_num)
    print('{} 고객의 {} 계좌가 등록되었습니다'
          .format(customer.get_name(),
                  acount_num,
                  ))

def show_customer():
    c_id = input('고객 번호 입력:')
    customer = search_customer(c_id)
    if customer is None:
        return

    print('{} 고객님 등급:{} 총금액:{} 계좌정보:{}'
          .format(customer.get_name(),
                  customer.get_rat(),
                  customer.get_total_amount1(),
                  customer.get_all_accounts()
                  ))

def deposit():
    c_id = input('고객 번호 입력:')
    customer = search_customer(c_id)
    if customer is None:
        return
    acount_num = input('계좌번호 입력:')
    amount = input('입금할 금액 입력:')
    customer.add_amount(acount_num, int(amount))

def withdraw():
    print('고객 번호 입력:')
    c_id = input()
    customer = search_customer(c_id)
    if customer is None:
        return

    print('계좌번호 입력:')
    acount_num = input()
    print('출금할 금액 입력:')
    amount = input()
    customer.sub_amount(acount_num, int(amount))

def txt_to_customers():
    input_path = input('텍스트 파일 불러오기 경로 입력') # ./customers.txt
    try : # 오류 모니터링 영역
        f = open(input_path, 'r', encoding='utf-8')
    except : #???
        print(f'텍스트파일 고객 정보 불러오기 실패')
    else : # 코드 실행 영역
        data_rlines = f.readlines()
        num = 0
        for line in data_rlines:
            num += 1
            c_name, c_id = line.split()
            c = Customer(c_name, c_id)
            customer_list.append(c)
        print(f'고객 정보 {num}명 텍스트 파일 불러오기 성공')
        f.close()

import json
def customer_to_json():
    # 딕셔너리 구조 만들기
    # [{'name': '최원칠', 'c_id': '123', 'rating': 'bronze',
    # 'total_amount': 0, 'accounts': {'1251': 0}} ...],
    all_data = []
    for c in customer_list:
        c_dict = {
            'name': c.get_name(),
            'c_id': c.get_id(),
            'rating': c.get_rat(),
            'total_amount': c.get_total_amount1(),
            'accounts': c.get_all_accounts()
        }
        all_data.append(c_dict)
        print(c_dict)
    print(all_data)

    # 입력받은 Path로 json저장하기
    # 계좌 생성 옵션 최소 2번 이상 실행 후, 해당 계좌가 들어간 상태로 저장되어야함
    try : # 오류 모니터링 영역
        input_path = input('json 파일 저장 경로 입력')
    except:  # ???
        print(f'고객 정보 json파일 저장 실패')
    else:  # 코드 실행 영역
        file = open(input_path, 'w')
        json.dump(all_data, file)
        print(f'고객 정보 json파일 저장 성공')
        file.close()



if __name__ == '__main__':
    customer_list = []

    print('1 - 고객 생성')
    print('2 - 계좌 생성')
    print('3 - 입금')
    print('4 - 출금')
    print('5 - 생성된 고객 리스트 출력')
    print('6 - 고객 정보 출력')
    print('7 - 고객 텍스트 파일 불러오기')
    print('8 - 고객 데이터 json 파일 저장')

    while True:
        print('--'*30)
        input1 = input('옵션을 입력해 주세요')
        if input1 == '1':
            customer = create_customer()
            customer_list.append(customer)
        elif input1 == '2':
            create_acount()
        elif input1 == '3':
            deposit()
        elif input1 == '4':
            withdraw()
        elif input1 == '5':
            show_list()
        elif input1 == '6':
            show_customer()
        elif input1 == '7':
            txt_to_customers()
        elif input1 == '8':
            customer_to_json()
        else:
            print('존재하지 않는 명령어 입니다.')
