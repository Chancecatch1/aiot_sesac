import pandas as pd
############ 참고 사항 ############
# 아키텍처 구성 #
## Backend : 클래스
## Backend : 스토어 모듈
## Frontend : 실행(main) 프로세스

# 데이터 구조 #
## aistore2 딕셔너리 -> aistore3 데이터프레임
## 최초에 데이터 베이스(csv임포트 데이터프레임)를 구성하고 시작
## aistore2와 달리 스토어모듈에서 인스턴스 목록이 아닌 데이터베이스(데이터프레임)에서 데이터를 가져와서 처리
## 상점/재고 데이터 변동 시 데이터베이스(s_df, iv_df)를 수시로 업데이트 필요
### 해당 코드에서는 클래스(set_product, buy_product)에서 iv_df만 업데이트함
## s_df는 업데이트 불필요(유일한 수정함수인 스토어모듈 create_store에서 s_df에 직접 추가함)
### 시리즈는 값이 하나일지라도 값과 비교할 수 없다. 차원을 낮추고 시도하자(squeeze)
### iv_df 인덱스 주의(s_df와 p_df만 id가 인덱스)

# 인스턴스 활용 #
## 스토어 인스턴스는 스토어 검색시에만 생성하며, '일부'다른 함수들이 스토어 검색 함수를 통해 인스턴스를 반환받는 구조
## 스토어 인스턴스를 반환받은 함수의 경우 필터링된 고객 데이터를 사용 가능(고객 쿼리 불필요)

# 오류/경고 처리 #
## df.update함수는 left조인만 가능하고 시스템에서 권장하지 않아 FutureWarning 발생, 무시해도 무방
## SettingWithCopyWarning는 무시해도 무방 : Chained Assignement(연쇄할당) 경고
pd.set_option('mode.chained_assignment',  None) # 경고 숨김 코드이지만 권장하진 않음

# 예외 처리 케이스 #
## 입력 값의 데이터 (존재 유무, 타입, 음수 여부) 체크

############ 클래스 정의 ############
class AiStore:

    def __init__(self, name, s_id, locate, products_num, inventory):
        self.name = name # value
        self.s_id = s_id # value
        self.locate = locate # value
        self.products_num = products_num # value
        self.inventory = inventory # dataframe

    def set_product(self, p_id, count, price):
        # 상품아이디에 해당하는 재고데이터프레임에 재고 추가및 가격 재등록
        iv_pids = self.inventory['p_id']
        # if in 을 사용하기위해선 시리즈를 배열로 바꿔야할것 문서 참고
        if p_id in list(iv_pids) : # 등록 상품 있을 경우
            pid_ivdf = self.inventory[iv_pids == p_id]
            pid_ivdf['count'] += count # 재고는 추가
            pid_ivdf['price'] = price # 가격은 덮어쓰기
            # iv_df는 인덱스가 없어 loc을 쓸수 없음.
            # 쿼리를 하되, 쿼리는 복사본이므로 반드시 원본을 업데이트 해주자.
            self.inventory.update(pid_ivdf)
            # print(self.inventory)
            print('기 등록상품 업데이트 성공')
        else : # 등록 상품 없을 경우
            # 상품아이디가 없다면 재고 및 가격 입력하여 재고 데이터프레임에 추가(append 또는 concat 함수사용)
            new_iv = pd.DataFrame([{'p_id': p_id, 'count': count, 'price': price}])
            self.inventory.update(new_iv)
            # self.inventory = pd.concat([self.inventory, new_ivdf])
            # self.inventory.loc[p_id, ['count', 'price']] = count
            # self.inventory.loc[p_id, 'price'] = price
            print('미 등록상품 추가 성공')

        # 데이터 업데이트
        self.update_data(s_df, iv_df)
        # try 문사용 가능, 쿼리후 개수로 파악 가능   ??

    def buy_product(self, p_id, buy_count, buy_amount):
        iv_pids = self.inventory['p_id']
        if p_id in list(iv_pids):  # 등록 상품 있을 경우
            pid_ivdf = self.inventory[iv_pids == p_id].squeeze() # 시리즈로 변환
            # 시리즈는 값이 하나일지라도 값과 비교할 수 없으므로, 차원을 더 낮춰주자.
            # self.inventory[iv_pids == p_id]['count'] 는 시리즈이기 떄문
            print(type(pid_ivdf['count']))
            # 재고 수량 체크
            if pid_ivdf['count'] >= buy_count : # 재고 충분시
                # 해당 물품의 가격과 개수로 전체 지불 해야할 금액 산출
                need_amount = abs(buy_count * pid_ivdf['price'])
                changes = abs(need_amount - buy_amount)
                # 지불 금액 체크
                if buy_amount >= need_amount : # 금액 충분시
                    # 재고데이터프레임의 해당 상품의 개수를 차감
                    pid_ivdf['count'] -= buy_count  # 재고 차감
                    self.inventory.update(pid_ivdf)
                    # 입력금액 에서 지불해야할 금액 차이로 ‘잔돈은 {금액차} 입니다’ 출력
                    print(f'구매 성공! 잔돈은 {changes} 입니다')
                else : # 금액 부족 시
                    print(f'구매 실패! 금액이 {changes}만큼 부족합니다.')
            else : # 재고 부족시
                print(f'구매 실패! 재고가 {abs(pid_ivdf["count"]- buy_count)}만큼 부족합니다')
        else : # 등록상품 부재 시
            print('구매 실패! 상품이 존재하지 않습니다')

        # 데이터 업데이트
        self.update_data(s_df, iv_df)

    def get_name(self):
        return self.name

    def get_id(self):
        return self.s_id

    def get_locate(self):
        return self.locate

    def get_products_num(self):
        return self.products_num

    def get_inventory(self):
        return self.inventory

    def show_products(self, p_df):
        # 가지고 있는 재고데이터프레임의 모든 물품 정보를 출력(상품명 포함되어야함)
        # '상품명:{} - 가격:{} (재고{}) id:{}’ 문자열이 등록상품 개수만큼 나와야함
        for p in self.inventory.iloc:
            print(f'상품명:{ p_df.loc[p["p_id"], "product"]} - 가격:{p["price"]} (재고{p["count"]}) id:{p["p_id"]}')

    def get_price(self, p_id):
        product =  self.inventory[self.inventory['p_id'] == p_id]
        if len(product) == 0:
            return None

        return product['price'].iloc[0]

    def update_data(self, s_df, iv_df):
        # 클래스내부에 변경된 데이터들을 각각 데이터프레임에 업데이트
        iv_df.update(self.inventory)
        # for idx in self.inventory.index:
        #     iv_df.loc[idx] = self.inventory.loc[idx]
        print('인벤토리 데이터프레임 업데이트 성공')

        # 현재 구조에서는 s_df 업데이트 불필요
        s_df[self.s_id] = {'s_id': self.s_id,
                           'name': self.name,
                           'locate': self.locate,
                           'products_num': self.products_num}
        print('스토어 데이터프레임 업데이트 성공')

############ 커머스 모듈 프로세스 ############
def create_store():
    # 스토어 데이터프레임에 입력받은 스토어아이디를 인덱스로 스토어데이터 추가
    # 스토어데이터는 스토어데이터프레임의 컬럼에 맞게 구성(딕셔너리추천)
    s_name = input('스토어 이름 입력: ')
    s_id = input('스토어 번호 입력: ')
    locate = input('스토어 위치 입력: ')
    s_df.loc[s_id] = {'s_id': s_id, 'name': s_name, 'locate': locate, 'products_num': 0}
    print(f'{s_name} 스토어가 생성 되었습니다.')

def show_list():
    # 모든 스토어의 정보를 다음과 같이 출력
    # ‘스토어이름:{} 스토어 아이디:{} 스토어 위치:{} 등록상품:{}’
    # 한줄씩 출력해야하니 반복문으로 꺼내야하고, 스토어별 상품종류가 복수이므로 반복문을 한번 더 써야함
    for store in s_df.iloc:
        store_pid = iv_df.loc[iv_df['s_id'] == store['s_id'], 'p_id']  # 스토어별 등록상품 아이디
        print(f'스토어 이름:{store["name"]}, 스토어 아이디:{store["s_id"]}, 스토어 위치:{store["locate"]},\
 등록상품:{list(p_df.loc[store_pid, "product"])}')

def search_store(s_id):
    if s_id in s_df['s_id']:
        # 스토어가 있다면 스토어 데이터프레임 에서 해당 스토어아이디에 해당하는 데이터 추출
        store = s_df.loc[s_id]
        # 스토어가 있다면 재고 데이터프레임 에서 해당 스토어아이디에 해당하는 재고데이터프레임 추출
        inventory = iv_df[iv_df['s_id'] == s_id]
        # 추출한 데이터로부터 AiStore인스턴스 생성후 반환(리턴)
        store_instance = AiStore(store['name'], store['s_id'], store['locate'], store['products_num'], inventory)
        return store_instance
    else: # 스토어가 없다면 None 반환
        print('해당 스토어를 찾지 못했습니다.')
        return None

def show_store():
    # 입력받은 스토어 아이디로 search_store 를 통해 스토어 인스턴스를 받음
    # 스토어 이름:{} 스토어 아이디:{} 스토어 위치:{} 등록상품:{}’ 출력
    s_id = input('스토어 아이디 입력: ')
    store_instance = search_store(s_id)
    if store_instance is None:
        return print(f'{s_id} 스토어 아이디가 유효하지 않습니다..')

    print(f'스토어이름:{store_instance.get_name()} 스토어 아이디:{store_instance.get_id()} 스토어 위치:{store_instance.get_locate()}\
\n등록상품:{list(p_df.loc[store_instance.get_inventory()["p_id"], "product"])}')

    # 스토어의 show_products 함수 실행
    store_instance.show_products(p_df)

def buy():
    s_id = input('스토어 번호 입력: ')
    # 입력받은 스토어 아이디로 search_store 를 통해 스토어 인스턴스를 받음
    store_instance = search_store(s_id)

    if store_instance is None:
        return print(f'{s_id} 스토어 아이디가 유효하지 않습니다..')

    # 스토어의 show_products 함수 실행
    store_instance.show_products(p_df)

    # 스토어가 있으면 상품아이디, 개수, 지불금액 입력받음 (옵션으로 필요 지불금액 출력)
    # get_price 함수로 필요가격 출력
    p_id = input('상품 아이디 입력:')
    count = int(input('구매 개수 입력: '))
    print(f'필요한 지불 금액은 {store_instance.get_price(p_id) * count}입니다.')

    # buy_product 함수로 물품 구매 진행
    amount = int(input('가격 입력: '))
    store_instance.buy_product(p_id, count, amount)

def manager_product():
    s_id = input('스토어 아이디 입력: ')
    # 입력받은 스토어 아이디로 search_store 를 통해 스토어 인스턴스를 받음
    store_instance = search_store(s_id)
    if store_instance is None:
        return print(f'{s_id} 스토어 아이디가 유효하지 않습니다..')
    # '등록 가능 상품 {상품데이터프레임}' 을 출력(print)
    print(f'등록 가능 상품 {p_df}')
    # 상품아이디, 개수, 가격 입력받음
    p_id = input('상품 아이디 입력: ')
    count = int(input('재고 개수 입력: '))
    price = int(input('상품 가격 입력: '))
    # set_product 함수 이용하여 해당 스토어 재고 및 가격 수정
    store_instance.set_product(p_id, count, price)


import json
def products_counts():
    # 재고데이터프레임에 상품데이터프레임을 상품아이디(’p_id’) 기준으로 결합(merge 또는 join)
    # 결합된 데이터프레임에서 상품이름과 상품개수만을 부분 데이터프레임으로 추출
    # 추출한 부분데이터 프레임에서 상품이름을 그룹화하여 상품이름별 총 재고개수를 데이터프레임으로 출력
    piv_df = p_df.join(iv_df.set_index('p_id'))
    p_group_df = piv_df[['product', 'count']].groupby('product').sum('count')
    print(p_group_df)


############ 실행 프로세스 ############
if __name__ == '__main__':
###### 데이터 준비 ######
    # 'stores.csv’ 를 불러와서 s_df에 데이터프레임으로 할당
    # set_index 함수를 활용해 ‘s_id’ 컬럼을 인덱스로(’s_id’ 컬럼유지)
    s_df = pd.read_csv('stores.csv')
    s_df.set_index('s_id', drop=False, inplace=True)  # drop=false는 인덱스로 활용된 칼럼 유지
    print(s_df.dtypes,'상점 데이터 불러오기 성공')

    # ‘products.csv’ 를 불러와서 p_df에 데이터프레임으로 할당
    # set_index 함수를 활용해 ‘p_id’ 컬럼을 인덱스로(’p_id’ 컬럼삭제)
    p_df = pd.read_csv('products.csv')
    p_df.set_index('p_id', inplace=True)
    print(p_df.dtypes, '상품 데이터 불러오기 성공')

    # ‘inventory.csv’ 를 불러와서 iv_df에 데이터프레임으로 할당
    # 순서 인덱스 유지
    iv_df = pd.read_csv('inventory.csv')
    print(iv_df.dtypes, '재고 데이터 불러오기 성공')

###### 스토어 메뉴 안내 ######

    print('1 - 스토어 생성')
    print('2 - 스토어 리스트 출력')
    print('3 - 스토어 정보 출력')
    print('4 - 상품 구매')
    print('5 - 상품 관리')
    print('6 - csv 파일로 스토어, 재고현황 파일 출력')
    print('7 - 상품명별 전체 재고 개수 출력')
    print('0 - 커머스 종료')

    while True:
        print('--'*30)
        input1 = input('옵션을 입력해 주세요: ')
        if input1 == '1': # 스토어 생성
            create_store()
        elif input1 == '2': # 스토어 리스트 출력
            show_list()
        elif input1 == '3': # 스토어 정보 출력
            show_store()
        elif input1 == '4': # 상품 구매
            buy()
        elif input1 == '5': # 상품 관리
            manager_product()
        elif input1 == '6': # csv 파일로 스토어, 재고현황 파일 출력
            # 6번 옵션 실행시 스토어데이터프레임 및 재고데이터프레임 csv 파일로 각각 저장
            # 저장시 인덱스는 빼고 저장
            s_df.to_csv('store_output.csv', index=False)
            print('store_output.csv 스토어 데이터 저장 완료')
            iv_df.to_csv('inventory_output.csv', index=False)
            print('inventory_output.csv 재고현황 데이터 저장 완료')
        elif input1 == '7': # 상품별 전체 재고 개수 출력
            products_counts()
        elif input1 == '0':
            print('커머스 종료')
            break
        else:
            print('존재하지 않는 명령어 입니다.')