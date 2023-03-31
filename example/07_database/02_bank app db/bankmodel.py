from sqlalchemy import Column, Integer, String, ForeignKey
from database import Base, db_session

#### 데이터베이스 테이블 정의 ####
class Customer(Base):
    __tablename__ = 'customers'
    c_id = Column(String(20), primary_key=True)
    name = Column(String(50))
    account_num = Column(String(20), unique=True)
    total_amount = Column(Integer)
    rat = Column(String(20))

    def __init__(self, c_id, name):
        self.c_id = c_id
        self.name = name
        self.account_num = 0
        self.total_amount = 0
        self.rat = 'normal'

    def add_account(self):
        self.account_num += 1

    def add_amount(self, amount):
        self.total_amount += amount
        self.set_rat()

    def sub_amount(self, amount):
        self.total_amount -= amount
        self.set_rat()

    def set_rat(self):
        if self.total_amount > 100000:
            rat = 'vvip'
        elif self.total_amount > 10000:
            rat = 'vip'
        elif self.total_amount > 1000:
            rat = 'gold'
        elif self.total_amount > 100:
            rat = 'silver'
        else:
            rat = 'bronze'

        self.rat = rat

    def __repr__(self):
        return '<{}, {}, {}, {}, {}>'.format(self.c_id, self.name, self.account_num, self.total_amount, self.rat)

class Accounts(Base):
    __tablename__ = 'accounts'
    a_id = Column(String(20), primary_key=True)
    c_id = Column(String(50), ForeignKey(Customer.c_id))
    amount = Column(Integer)

    def __init__(self, a_id, c_id):
        self.c_id = c_id
        self.a_id = a_id
        self.amount = 0

    def add_amount(self, amount):
        self.amount += amount

    def sub_amount(self, amount):
        self.amount -= amount

    def __repr__(self):
        return '<{}, {}, {}>'.format(self.a_id, self.amount, self.c_id)


#### 뱅킹 모듈 ####
def show_list():
    result = []
    # 데이터베이스의 모든 고객을 쿼리
    customers = Customer.query.all()
    for c in customers:
        result.append([c.name, c.c_id ,c.rat, c.total_amount])
    print(result)
    return result

def show_customer(customer):
    customer_view = [[customer.name,
                     customer.c_id,
                     customer.rat,
                     customer.total_amount]]
    # 고객 아이디에 해당하는 계좌인스턴스 쿼리
    accounts = Accounts.query.filter(Accounts.c_id == customer.c_id)
    account_view = []
    for account in accounts:
        account_view.append(account)

    return customer_view, account_view

def create_customer(c_id , c_name):
    if db_session.get(Customer, c_id) is None:
        # 고객 인스턴스 생성 -> 데이터베이스에 add -> commit
        c = Customer(c_id = c_id, name = c_name)
        db_session.add(c)
        db_session.commit()
        return print('고객 등록 성공')
    else:
        return print(f'생성 실패, {c_id} 이미 등록된 고객 아이디 입니다.')

def create_account(customer: Customer, a_id): # customer : Customer 은 app.py으로 부터 ORM객체를 받은 것
    # 고객 인스턴스에 함수를 활용하여 account_num 증가
    if db_session.get(Accounts, a_id) is None:
        # 고객 인스턴스 생성 -> 데이터베이스에 add -> commit
        a = Accounts(a_id = a_id, c_id = customer.c_id)
        # 계좌 인스턴스 생성 -> add
        db_session.add(a)
        # 최종 commit
        db_session.commit()
        customer.account_num += 1
        return print('계좌 등록 성공')
    else:
        return print(f'생성 실패, {a_id} 이미 등록된 계좌 번호 입니다.')

def deposit(customer: Customer, account : Accounts, amount: int):
    if customer is None:
        return print(f'고객 정보가 유효하지 않습니다.')
    elif account is None:
        return print(f'계좌 정보가 유효하지 않습니다.')
    else:
        # 계좌 인스턴스의 함수를 활용하여 계좌금액 증가
        account.amount += amount
        # 고객 인스턴스의 함수를 활용하여 총액 증가
        customer.total_amount += amount
        # 최종 commit
        db_session.commit()
        return print('입금 성공')

def withdraw(customer: Customer, account : Accounts, amount: int):
    # 계좌 인스턴스의 금액이 출금 금액 보다 클경우만
    if account.amount >= amount :
        # 계좌 인스턴스의 함수를 활용하여 계좌금액 차감
        account.amount -= amount
        # 고객 인스턴스의 함수를 활용하여 총액 차감
        customer.total_amount -= amount
        # 최종 commit
        db_session.commit()
        print('출금 성공')
    else :
        return print('출금 실패, 잔액이 부족합니다.')