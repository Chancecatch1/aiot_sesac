## 참고 사항 ##
# get_menu 클래스 메서드는 해당 스토어의 상품 정보로서 manage에서 사용됨
# get_product 클래스 메서드는 해당 상품아이디의 상품 정보로서 buy에서 사용됨
# get_products 함수는 데이터베이스에 존재하는 전체 상품 재고 정보로 manage에서 사용됨
# ㄴ 이름이 헷갈리니까 get_products_catalog로 수정함
# 데이터베이스 업데이트는 set_product()와 buy_product()와 함께 app에서 사용하자

import pandas as pd

s_df = pd.read_csv('./static/stores.csv')
s_df = s_df.set_index('s_id', drop=False)
p_df = pd.read_csv('./static/products.csv')
iv_df = pd.read_csv('./static/inventory.csv')

#### 클래스 정의 ####
class AiStore:

    def __init__(self, name, s_id, locate, products_num, inventory):
        self.name = name
        self.s_id = s_id
        self.locate = locate
        self.products_num = products_num
        self.inventory = inventory

    def add_product(self, p_id, count, price, idx):
        n_product = {'p_id': p_id, 'count': count, 'price': price, 's_id': self.s_id}
        self.inventory.loc[idx] = n_product # 마지막 행에 상품 등록
        self.products_num = len(self.inventory)
        return n_product

    def set_product(self, p_id, count, price):
        if p_id not in self.inventory['p_id']:
            return print('없는 제품 아이디 입니다. 하단 카탈로그를 참고해주세요.')
        else:
            # 해당하는 상품의 가격과 수량 값 변경
            product =  self.inventory[self.inventory['p_id'] == p_id]
            product['count'] += count
            product['price'] = price
            self.inventory.update(product) # 데이터베이스 업데이트 아님

    def is_product(self, p_id):
        # 해당 id의 상품의 db유무 체크
        product = self.inventory[self.inventory['p_id'] == p_id]
        if len(product) == 0:
            return False
        else:
            return True

    def buy_product(self, p_id, count):

        product =  self.inventory[self.inventory['p_id'] == p_id]
        product = product[product['count'] > count]
        if len(product) == 0: # 금액 부족 케이스
            print('금액이 부족합니다.')
            return
        # 금액 충분 케이스
        product['count'] -= count
        self.inventory.update(product) # 데이터베이스 업데이트 아님

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

    def get_menu(self): # 해당 스토어의 전체 상품 정보(재고 현황)
        # manage에서 목록으로 받으므로 리스트로 생성하자
        # 인벤토리의 상품들을 메뉴 리스트에 추가
        menu = []
        for product in self.inventory.iloc():
            menu.append({'p_name': product['product'],
                         'price': int(product['price']),
                         'count': int(product['count']),
                         'p_id': product['p_id']})
        return menu

    def get_product(self, p_id): # 해당하는 상품의 상품정보
        # p_id 해당하는 상품 반환 인벤토리에서 쿼리 후 .iloc[0] 을 통해 상품 가져올것
        # 쿼리 결과물이 df인데 html템플릿에서 product.price같은 메서드를 사용하려면 딕셔너리(시리즈)변환 필
        product = self.inventory[self.inventory['p_id'] == p_id].iloc()[0]

        return {'p_name': product['product'],
                'price': int(product['price']),
                'count': int(product['count']),
                'p_id': product['p_id']}


#### 스토어 모듈 ####

def create_store(s_id, s_name, locate):

    store = {'s_id': s_id, 'name': s_name,
             'locate': locate,
             'products_num': 0,}

    s_df.loc[s_id] = store
    print('{} 스토어가 생성 되었습니다.'.format(store['name']))


def show_list(s_id = None):
    if s_id is None:
        return s_df.to_dict('records')
    else:
        return s_df[s_df['s_id'] == s_id].to_dict('records')

def search_store(s_id):
    if s_id in s_df.index:
        store = s_df.loc[s_id]
        # 인벤토리에서 s_id에 해당하는 상품 쿼리 후 p_df 와 p_id를 기준으로 머지 (인벤토리가 왼쪽이 되야함)
        # 문제중 머지하는 부분을 join 함수로 해주셔야 인덱스가 변하지 않아서 데이터가 안섞입니다.
        # merge함수는 반환된 결과의 인덱스를 초기화 시키기때문에 iv_df을 업데이트하는순간 기존 데이터를 덮어 씌우게 되서 문제가 생깁니다.
        # inventory.join(p_df.set_index(’p_id’), on = ‘p_id’) 조인은 인덱스를 기준으로 잡고 데이터를 합치기때문에 반환된 결과가 인덱스가 초기화 되지 않습니다
        inventory = iv_df[iv_df['s_id'] == s_id]
        inventory = inventory.join(p_df.set_index('p_id'), on = 'p_id')
        store_instance = AiStore(store['name'],
                       store['s_id'],
                       store['locate'],
                       store['products_num'],
                       inventory)
        update(store_instance)
        return store_instance
    else:
        print('스토어를 찾지 못했습니다.')
        return None

def get_products_catalog():
    # 각 col을 key로 각row를 dictionary로 변환하여 리스트로 기록
    return p_df.to_dict('records')

def set_product(store, p_id, price, count):

    if store.is_product(p_id):
        store.set_product(p_id, int(count), int(price))
    else:
        idx = len(iv_df)
        # df의 기본 idx는 0부터 시작하고, 1부터 시작하는 len길이 값으로 인덱싱하면 마지막에 데이터 입력할 수 있다.
        n_product = store.add_product(p_id, int(count), int(price), idx)
        iv_df.loc[idx] = n_product

def update(store: AiStore):
    # s_df는 s_id 인덱스이므로 인덱싱을 통한 데이터 입력 가능
    s_df[store.get_id()] = {'s_id': store.get_id(),
                       'name': store.get_name(),
                       'locate': store.get_locate(),
                       'products_num': store.get_products_num(), }
    # Iv_df는 인덱스가 없으니 update함수로 변경/추가된 데이터를 모두 수정해주자
    iv_df.update(store.get_inventory())