from flask import Flask, request, render_template, redirect, url_for, session
import sys
from aistoremodel import *
import datetime
from wtforms import Form, StringField, PasswordField, TextAreaField, validators

app = Flask(__name__)
app.config['SECRET_KEY'] = 'aiot'

@app.before_request # http요청이 들어올때마다 실행(http요청 핸드러)
def before_request():
    session.permanent = True
    app.permanent_session_lifetime = datetime.timedelta(minutes=5)
    session.modified = True

@app.route("/", methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route("/sregister", methods=['POST', 'GET'])
def sregister():
    if request.method == 'POST':
        s_id = request.form['sId']
        s_name = request.form['sName']
        locate = request.form['locate']
        create_store(s_id, s_name, locate)
        return redirect('/')

    return render_template('sregister.html')

@app.route("/stores", methods=['POST', 'GET'])
def stores():
    if request.method == 'POST':
        s_id = request.form['sId']

        return render_template('stores.html', stores = show_list(s_id))

    return render_template('stores.html', stores = show_list())

@app.route("/manage/<s_id>", methods=['POST', 'GET'])
def manage(s_id = 'nan'):
    if request.method == 'POST':
        if s_id == 'nan':
            s_id = request.form['sId']
            # Products 전체 쿼리 (리스트)
            products = Products.query.all()
            return render_template('manage.html',
                                    s_id = s_id,
                                    inventory=get_menu(s_id),
                                    products=products
                                    )
        else:
            p_id = request.form['pId']
            price = request.form['price']
            count = int(request.form['count'])
            set_product(s_id, p_id, price, count)
            # Products 전체 쿼리 (리스트)
            products = Products.query.all()
            return render_template('manage.html',
                                    s_id = s_id,
                                    inventory = get_menu(s_id),
                                    products = products
                                   )

    return render_template('manage.html', s_id = s_id, )

@app.route("/board/<s_id>", methods=['POST', 'GET'])
def board(s_id = 'nan'):
    # buy 페이지에서 뒤로가기 버튼 또는 메인페이지 버튼(nav바) 클릭시 세션에 이전 데이터가 저장되어있을수 있음
    # 따라서 다시 buy 페이지로 가기전에 미리 세션 초기화
    if 'count' in session:
        del session['count']
    if 'buy_product' in session:
        del session['buy_product']

    if request.method == 'POST':
        s_id = request.form['sId']
        # aistore4_app2와 달리 get_menu가 클래스 메서드가 아니므로 불필요
        # ai_store = db_session.get(AiStore, s_id)
        # return render_template('board.html', s_id = s_id, menu = get_menu(ai_store))
        return render_template('board.html', s_id=s_id, menu=get_menu(s_id))
    if s_id != 'nan':
        # 위와 상동
        # ai_store = db_session.get(AiStore, s_id)
        # return render_template('board.html', s_id=s_id, menu = get_menu(ai_store))
        return render_template('board.html', s_id=s_id, menu = get_menu(s_id))

    else:
        return render_template('board.html', s_id=s_id,)

@app.route("/buy/<s_id>/<p_id>", methods=['POST', 'GET'])
def buy(s_id, p_id):
    if 'buy_product' not in session:
        inventory = db_session.get(Inventory,(p_id,s_id))
        # aistore4_app2와 달리 buy_product 클래스 메서드가 없으므로, db에서 가져오자.
        session['buy_product'] = {'p_name': inventory.product.name, 'price': inventory.price}

    # 기본 알림값 초기화(미실행)
    alert = False

    # 세션 카운트값 초기화
    if 'count' not in session:
        session['count'] = 1

    # 값이 들어올 경우, 즉 구매페이지를 통한 접근 케이스
    if request.method == 'POST':
        if request.form.get('plus') == '+':
            session['count'] +=1

        elif request.form.get('sub') == '-':
            if session['count'] > 1:
                session['count'] -=1

        else: # 위 +/- 버튼이 아니므로 구매 버튼이 됨,
            if buy_product(p_id, s_id, session['count']):
                # 불린값 반환, true 구매 성공시 페이지 리다이렉트
                return redirect(url_for('board', s_id = s_id))
            else: # 구매 실패시 alert값을 참으로 변경하여 템플릿에서 알림 동작(페이지 새로 고침 안해도 됨)
                alert = True
    # 최초 인덱스를 통한 접속 케이스
    return render_template('buy.html',
                           s_id=s_id, p_id = p_id,
                           product = session['buy_product'],
                           count = session['count'],
                           alert = alert
                           )

# http요청 결과가 브라우저에 응답한 다음 실행
@app.teardown_request
def shutdown_session(exception=None):
    db_session.remove()



if __name__ == '__main__':
    app.run('0.0.0.0',9999, debug=True)