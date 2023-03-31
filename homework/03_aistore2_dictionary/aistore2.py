#### 클래스 영역 ####
class AiStore:
    def __init__(self, name, s_id, locate):
        self.name = name
        self.s_id = s_id
        self.locate = locate
        self.products = {'커피' : 10}
        self.prices = {'커피' : 2000}

    def set_product(self, product, count, price):
        if product in self.products :
            self.products[product] += int(count)
            self.prices[product] = int(price)
        else :
            self.products[product] = int(count)
            self.prices[product] = int(price)

    def buy_product(self, product, buy_count, buy_price):
        if self.products[product] < 0: # 재고 부족시
            print(f'구매 실패, 재고가 {abs(self.products[product] - buy_count)}개 부족합니다.')
        else : # 재고 충분시
            need_amount = self.prices[product] * buy_count
            change = abs(buy_price - need_amount)
            if buy_price < need_amount: # 금액 부족시
                print(f'구매 실패, 금액이 {change}만큼 부족합니다.')
            else : # 금액 충분시 결제 완료
                self.products[product] -= buy_count
                print(f'구매 성공, 잔돈은 {change}원이고, 남은 재고는 {self.products[product]}입니다.')

    def get_name(self):
        return self.name

    def get_id(self):
        return self.s_id

    def get_locate(self):
        return self.locate

    def get_products(self):
        return self.products

    def get_prices(self):
        return self.prices

#### 프로그램 영역 ####
def create_store():
    s_name = input('스토어 이름 입력: ')
    s_id = input('스토어 아이디 입력: ')
    locate = input('스토어 위치 입력: ')
    store = AiStore(s_name, s_id, locate)
    print('{} 스토어가 생성 되었습니다.'.format(store.get_name()))
    return store

def show_list():
    for store in store_list:
        print('스토어 이름:{} 스토어 아이디:{} 스토어 위치:{}'
              .format(store.get_name(),
                      store.get_id(),
                      store.get_locate()
                      ))

def search_store(s_id):
    for store in store_list:
        if store.get_id() == s_id:
            print('스토어 이름:{} 스토어 번호:{}'
                  .format(store.get_name(),
                          store.get_id()
                          ))
            return store
        # 포문을 계속 돌기 위해 else 생략
    print('스토어 아이디를 찾지 못했습니다.')
    return None

def show_store():
    s_id = input('스토어 아이디 입력: ')
    store = search_store(s_id)
    if store is None:
        return

    print('{} 스토어 재고 현황: {}'
          .format(store.get_name(),
                  store.get_products()
                  ))
    print('{} 스토어 가격 현황: {}'
          .format(store.get_name(),
                  store.get_prices()
                  ))

def buy():
    s_id = input('스토어 아이디 입력: ')
    store = search_store(s_id)
    if store is None:
        return

    product = input('상품 입력:')
    buy_count = int(input('구매 개수 입력: '))
    buy_price = int(input('가격 입력: '))
    # 옵션 '총 가격은 {} 입니다.' 출력
    print(f'총 구매 가격은 {buy_count * buy_price} 입니다.')
    store.buy_product(product, buy_count, buy_price)

def manager_product():
    s_id = input('스토어 번호 입력: ')
    store = search_store(s_id)
    if store is None:
        return

    product = input('상품 입력: ')
    count = input('재고 개수 입력: ')
    price = input('상품 가격 입력: ')
    store.set_product(product, count, price)

def txt_to_store():
    input_path = input('txt파일 불러오기 경로 입력') # ./221112_stores.txt
    try : # 오류 모니터링 영역
        f = open(input_path, 'r', encoding='utf-8')
    # with open(input_path, 'r',  encoding='utf-8'0 as f: # with는 close선언 생략 가능
    except : #???
        print('스토어 정보 텍스트 파일 불러오기 실패')
    else : # 코드 실행 영역
        data_rlines = f.readlines()
        num = 0
        for line in data_rlines:
            num += 1
            s_name, s_id, s_locate = line.split()
            s = AiStore(s_name, s_id, s_locate)
            store_list.append(s)
        print(f'스토어 정보 {num}개 텍스트 파일 불러오기 성공')
        f.close()

import json
def store_to_json():
    all_data = []
    for store in store_list:
        store_dict = {
            'name': store.get_name(),
            's_id': store.get_id(),
            'products': store.get_products(),
            'price' : store.get_prices(),
            'locate' : store.get_locate()
        }
        all_data.append(store_dict)
    print(all_data)

    # 입력받은 Path로 json저장하기
    # 변경한 재고 및 가격 정보도 포함되어야 함
    try : # 오류 모니터링 영역
        input_path = input('json파일 저장 경로 입력')
    except:  # ???
        print(f'스토어 정보 json파일 저장 실패')
    else:  # 코드 실행 영역
        file = open(input_path, 'w',)
        json.dump(all_data, file, ensure_ascii=False)
        print(f'스토어 정보 json파일 저장 성공')
        file.close()

##### 실행 영역 #####
if __name__ == '__main__':
    store_list = []

    print('1 - 스토어 생성')
    print('2 - 스토어 리스트 출력')
    print('3 - 스토어 정보 출력')
    print('4 - 상품 구매')
    print('5 - 상품 관리')
    print('6 - txt 파일로 스토어 생성')
    print('7 - json 파일로 스토어 정보 출력')

    while True:
        print('--'*30)
        input1 = input('옵션을 입력해 주세요: ')
        if input1 == '1': # 스토어 생성
            store = create_store()
            store_list.append(store)
        elif input1 == '2': # 스토어 리스트 출력
            show_list()
        elif input1 == '3': # 스토어 정보 출력
            show_store()
        elif input1 == '4': # 상품 구매
            buy()
        elif input1 == '5': # 상품 관리
            manager_product()
        elif input1 == '6': # txt 파일로 스토어 생성
            txt_to_store()
        elif input1 == '7': # json 파일로 스토어 정보 출력
            store_to_json()
        else:
            print('존재하지 않는 명령어 입니다.')