import pandas as pd
############ 참고 사항 ############
# 데이터 구조 #
## bank2 딕셔너리 -> bank3 데이터프레임
## 최초에 데이터 베이스(csv임포트 데이터프레임)를 구성하고 시작
## bank2와 달리 각 뱅킹 모듈에서 인스턴스 목록이 아닌 데이터베이스(데이터프레임)에서 데이터를 가져와서 처리
## 고객/계좌 데이터 변동 시 데이터베이스(c_df, a_df)를 수시로 업데이트 필요

# 인스턴스 활용 #
## 고객 인스턴스는 고객 검색시에만 생성하고, '일부(중요)' 뱅킹 함수가 고객 검색 함수를 통해 고객 인스턴스를 반환
## 고객 인스턴스를 반환받은 함수의 경우, 필터링된 고객 데이터를 사용 가능(고객 쿼리 불필요)

# 오류/경고 처리 #
## df.update함수는 left조인만 가능하고 시스템에서 권장하지 않아 FutureWarning 발생, 무시해도 무방
## SettingWithCopyWarning는 무시해도 무방 : Chained Assignement(연쇄할당) 경고
pd.set_option('mode.chained_assignment',  None) # <==== 경고를 끈다

# 예외 처리 케이스 #
## 입력 값의 데이터 (존재 유무, 타입, 음수 여부) 체크 : 계좌 생성, 입금, 출금


############ 클래스 정의 ############
class Customer:

    def __init__(self, customer, accounts):
        self.customer = customer # series
        self.accounts = accounts # dataframe

    def add_account(self, a_id):
        if a_id in self.accounts.index:
            print('해당 계좌가 이미 존재합니다')
        else:
            # self.accounts 에 a_df 형태에 맞게 새로운 계좌 데이터를 할당
            # SWCW 오류 : 코드 동작하므로 무시해도 무방, df 슬라이싱이 뷰인지 수정인지 의도를 몰라 경고하는 것
            self.accounts.loc[a_id] = {'a_id': a_id, 'amount': 0, 'c_id': self.customer['c_id']}
            # self.customer 의 계좌개수를 +1
            self.customer['account_num'] += 1

    def add_amount(self, a_id, amount):
        if a_id in self.accounts.index:
            # self.accounts 에서 해당 계좌번호의 계좌금액에서 입금금액 추가
            self.accounts.loc[a_id, 'amount'] += amount
            # 원래 이미 고객 인스턴스 베이스로 a_id필터링된 데이터이므로 a_id인덱싱 불필요 하지만
            # self.accounts['amount']+= amount # 의 경우 SettingWithCopyWarning 경고를 발생함
            # SettingWithCopyWarning 경고는 df일부를 슬라이싱해서 수정할 경우 복사본만 수정할 지 원본도 수정할지 의도를 몰라 경고하는 것
            # 해당 코드는 df.copy()가 해당하지 않는 경우이므로 명시적인 .loc 인덱싱을 통해 해결하자.

            # self.customer 의 총합금액과 등급 업데이트
            # self.customer는 시리즈인데 왜 SWCW경고가 뜨는지 모르겠다. 해결이 복잡할 것 같으니 무시하고 넘어가자
            self.customer['total_amount'] = self.get_total_amount()
            self.customer['rat'] = self.get_rat()
        else:
            print('해당 계좌가 없습니다')

    def sub_amount(self, a_id, amount):
        if a_id in self.accounts.index:
            if self.accounts.loc[a_id, 'amount'] > amount: # 계좌 잔액 충분시
                # self.accounts 에서 해당 계좌번호의 계좌금액에서 입금금액 추가
                self.accounts.loc[a_id, 'amount'] -= amount
                # self.customer 의 총합금액과 등급 업데이트
                self.customer['total_amount'] = self.get_total_amount()
                self.customer['rat'] = self.get_rat()
            else: # 계좌 잔액 부족시
                print('계좌의 금액이 부족합니다.')
        else:
            print('해당 계좌가 없습니다')

    def update(self, c_df, a_df):
        # 클래스내의 수정된 고객시리즈 데이터를 고객데이터프레임에 업데이트
        c_df.update([self.customer])
        # c_df.loc[self.customer['c_id']] = self.customer

        # 클래스내의 수정된  데이터를 고객데이터프레임에 업데이트
        for a_idx in self.accounts.index: # 인덱싱으로 하나씩 삽입
            a_df.loc[a_idx] = self.accounts.loc[a_idx]
        # a_df = self.accounts # a_df 파라미터를 받는 변수로 사용 불가
        # a_df.update(self_accounts) # 데이터프레임 업데이트 함수는 left조인만 가능하므로 a_df에는 불가
        # a_df.loc[a_id] = self.accounts.loc[a_id] # a_id값을 모르므로 불가
        print('데이터베이스(c_df, a_df) 업데이트 완료')

    def get_total_amount(self):
        # 요청한 고객 객체가 갖고 있는 모든 계좌 금액의 총합을 반환
        # self.accounts[self.accounts['c_id'] == self.customer['c_id']]['amount'].sum()
        # 고객 검색시, 해당 고객의 계좌 정보만으로 객체가 생성됨, 필터링된 계좌정보이므로 쿼리할 필요없음
        return self.accounts['amount'].sum()
        # return self.accounts['amount'].sum()

    def get_rat(self): # 요청한 고객객체에 대한 등급을 반환
        total_amount = self.get_total_amount()
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
        return self.customer['name']

    def get_cid(self):
        return self.customer['c_id']

    def get_accounts(self):
        return self.accounts

    def get_customer(self):
        return self.customer

############ 뱅킹 프로세스 ############
def create_customer():
    c_name = input('고객 이름 입력: \n')
    c_id = input('고객 번호 입력: \n')
    # 고객 데이터프레임에 입력받은 고객아이디를 인덱스로 고객데이터 추가
    # 고객 데이터는 고객데이터프레임의 컬럼에 맞게 구성(딕셔너리 추천)
    customer = {'c_id': c_id,
             'name': c_name,
             'account_num': 0,
             'total_amount': 0,
             'rat': 'normal'}

    c_df.loc[c_id] = customer

    print('{} 고객이 생성 되었습니다.'.format(c_name))

def show_list():
    # 모든 고객의 고객 이름과 고객 아이디를 다음과 같이 출력
    # '고객이름:{} 고객번호:{}' {}안은 실제 데이터가 입력
    for c in c_df.iloc:
        print(f'고객이름 : {c["name"]}, 고객번호 : {c["c_id"]}')

def search_customer(c_id):
    # 고객이 있다면 고객데이터프레임에서 해당 고객아이디에 해당하는 시리즈 customer 변수에 할당
    if c_id in c_df.index:
        customer = c_df.loc[c_id]
        # 고객이 있다면 계좌데이터프레임 에서 해당 고객아이디에 해당하는 데이터프레임 account 변수에 할당
        account = a_df[a_df['c_id'] == c_id]
        # 데이터프레임은 기본적으로 복사본을 반환(리턴)
        # 해당 데이터로 고객 인스턴스 생성하여 반환
        customer_instance = Customer(customer, account)
        return customer_instance
    else: # 고객이 없다면 None 반환
        print('고객 번호를 찾지 못했습니다.')
        return None

def create_account():
    c_id = input('고객 번호 입력: \n')
    # search_customer 를 통해 고객을 찾고 새 계좌 생성
    customer = search_customer(c_id)
    if customer is None:
        return print(f'{c_id} 고객 번호가 유효하지 않습니다.')

    account_num = input('계좌 번호 입력: \n')
    customer.add_account(account_num)
    # 계좌, 고객 데이터프레임 업데이트(고객 클래스에 update활용)
    customer.update(c_df, a_df)
    # '{} 고객의 {} 계좌가 등록되었습니다' 출력
    print('{} 고객의 {} 계좌가 등록되었습니다'
          .format(customer.get_name(),
                  account_num,
                  ))

def show_customer():
    c_id = input('고객 번호 입력: \n')
    customer = search_customer(c_id)
    if customer is None:
        return print(f'{c_id} 계좌 번호가 유효하지 않습니다..')

    print('{} 고객님 등급:{} 총금액:{} 계좌정보: \n{}'
          .format(customer.get_name(),
                  customer.get_rat(),
                  customer.get_total_amount(),
                  customer.get_accounts()
                  ))

def deposit():
    c_id = input('고객 번호 입력: \n')
    # search_customer 를 통해 고객을 찾음
    customer = search_customer(c_id)
    if customer is None:
        return print(f'{c_id} 고객 번호가 유효하지 않습니다.')

    account_id = input('계좌 번호 입력: \n')
    amount = input('입금 금액 입력: \n')

    # 입력받은 계좌에 입력받은 금액 추가
    customer.add_amount(account_id, int(amount))
    # 데이터프레임 업데이트(고객 클래스에 update활용)
    customer.update(c_df, a_df)

def withdraw():
    c_id = input('고객 번호 입력: \n')
    # search_customer 를 통해 고객을 찾음
    customer = search_customer(c_id)
    if customer is None:
        return print(f'{c_id} 고객 번호가 유효하지 않습니다.')

    account_id = input('계좌 번호 입력: \n')
    amount = input('출금 금액 입력: \n')

    # 입력받은 계좌에 입력받은 금액 차감
    customer.sub_amount(account_id, int(amount))
    # 데이터프레임 업데이트(고객 클래스에 update활용)
    customer.update(c_df, a_df)

def ca_merge():
    # 고객데이터프레임에 계좌데이터프레임을 결합(merge)하여 ca_merge.csv’ 파일로 저장
    # 결합시 고객아이디(’c_id’) 를 기준으로 결합
    merge_df = c_df.join(a_df.set_index('c_id'))
    merge_df.to_csv('ca_merge.csv', index=False)
    print('ca_merge.csv 고객+계좌 통합 데이터 저장 완료')

def group_rat_count():
    # 고객등급별 고객 명수 출력. groupby함수 사용할것
    # 다음과 같은 데이터프레임으로 출력
    # 인덱스: 고객등급, 컬럼: 고객명수(’count’)
    result = c_df[['c_id', 'rat']].groupby('rat').count()
    result.rename(columns={'c_id': 'count'}, inplace=True)
    print(result)


############ 실행 프로세스 ############
if __name__ == '__main__':
###### 데이터 준비 ######
    c_df = pd.read_csv('bank3_customers.csv')  # index_col은 인덱스로 활용된 칼럼이 삭제됨
    c_df.set_index('c_id', drop=False, inplace=True)  # drop=false는 인덱스로 활용된 칼럼 유지
    print(c_df.dtypes, '고객 데이터 불러오기 성공')

    a_df = pd.read_csv('bank3_accounts.csv')
    a_df.set_index('a_id', drop=False, inplace=True)
    print(a_df.dtypes, '계좌 데이터 불러오기 성공')

    print('='*50)

###### 뱅킹 메뉴 안내 ######
    print('1 - 고객 생성')
    print('2 - 계좌 생성')
    print('3 - 입금')
    print('4 - 출금')
    print('5 - 생성된 고객 리스트 출력')
    print('6 - 고객 정보 출력')
    print('7 - csv 파일로 고객 정보 출력')
    print('8 - csv 파일로 계좌 정보 출력')
    print('9 - csv 파일로 고객 - 계좌 정보 출력')
    print('10 - 등급별 고객의 명수 출력')
    print('0 - 뱅킹 종료')

    while True:
        print('--'*30)
        input1 = input('옵션을 입력해 주세요 \n')
        if input1 == '1': # 고객 생성
            create_customer()
        elif input1 == '2': # 계좌 생성
            create_account()
        elif input1 == '3': # 입금
            deposit()
        elif input1 == '4': # 출금
            withdraw()
        elif input1 == '5': # 생성된 고객 리스트 출력
            show_list()
        elif input1 == '6': # 고객 정보 출력
            show_customer()
        elif input1 == '7': # csv 파일로 고객 정보 출력
            c_df.to_csv('customers_output.csv', index=False)
            print('customers_output.csv 고객 데이터 저장 완료')
        elif input1 == '8': # csv 파일로 계좌 정보 출력
            a_df.to_csv('accounts_output.csv', index=False)
            print('accounts_output.csv 계좌 데이터 저장 완료')
        elif input1 == '9': # csv 파일로 고객 - 계좌 정보 출력
            ca_merge()
        elif input1 == '10': # 등급별 고객의 명수 출력
            group_rat_count()
        elif input1 == '0':
            print('뱅킹 종료')
            break
        else:
            print('존재하지 않는 명령어 입니다.')