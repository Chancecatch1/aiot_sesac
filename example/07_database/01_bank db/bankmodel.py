from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from database import Base, db_session

class Customer(Base):
    __tablename__ = 'customers'
    c_id = Column(String(20), primary_key=True)
    name = Column(String(50))
    account_num = Column(Integer)
    total_amount = Column(Integer)
    rat = Column(String(20))

    def __init__(self, c_id, name):
        self.c_id = c_id
        self.name = name
        self.account_num = 0
        self.total_amount = 0
        self.rat = 'normal'

    def __repr__(self):
        return '<{}, {}, {}, {}, {}>'.format(self.c_id, self.name, self.account_num, self.total_amount, self.rat)


class Accounts(Base):
    __tablename__ = 'accounts'
    # a_id는 문자열에 primary_key 여야함
    # c_id는 문자열에 unique 여야함 ForeignKey
    # c_id는 ForeignKey 가됨 (Column 옵션에 두번째 인자로 ForeignKey(Customer.c_id) 추가)
    # amount는 숫자형 여야함
    a_id = Column(String(20), primary_key=True)
    c_id = Column(String(20), ForeignKey(Customer.c_id), unique= True )
    amount = Column(Integer)

    def __init__(self, a_id, c_id):
        # 생성시 a_id와 c_id는 인자를 통해 셀프 변수 생성
        # self.amounts는 0으로 초기화
        self.a_id = a_id
        self.c_id = c_id
        self.amount = 0

    def __repr__(self):
        return '<{}, {}, {}>'.format(self.a_id, self.amount, self.c_id)
