
import datetime
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from binance.client import Client
from binance import ThreadedWebsocketManager

# -----------------------------------------
# 1. DataHandler: 과거 데이터 확보 및 실시간 웹소켓 업데이트
# -----------------------------------------
class DataHandler:
    def __init__(self, client: Client, symbol: str, timestep: str):
        self.client = client
        self.symbol = symbol
        self.timestep = timestep  # 예: '1m', '3m', '5m', '10m', '30m', '60m', '1d'
        self.data = pd.DataFrame()
        self._initialize_data()
        self.twm = None  # 웹소켓 매니저

    def _initialize_data(self):
        """
        프로그램 시작 시 Binance API를 통해 과거 데이터를 확보합니다.
        """
        try:
            #klines = self.client.futures_klines(
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.timestep,
                limit=150
            )

            cols = ['open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'num_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            self.data = pd.DataFrame(klines, columns=cols)


            # open_time을 UTC로 변환 후 한국시간('Asia/Seoul')으로 변환
            self.data['open_time'] = pd.to_datetime(self.data['open_time'], unit='ms', utc=True)
            self.data['open_time'] = self.data['open_time'].dt.tz_convert('Asia/Seoul')

            self.data[['open', 'high', 'low', 'close', 'volume']] = self.data[['open', 'high', 'low', 'close', 'volume']].astype(float)
            self.data.set_index('open_time', inplace=True)
            print (self.data)
            assert False
            
        except Exception as e:
            print("과거 데이터 확보 중 오류 발생:", e)

    def start_realtime_socket(self, api_key: str, api_secret: str):
        """
        ThreadedWebsocketManager를 사용하여 실시간 kline 데이터를 받아옵니다.
        """
        self.twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
        self.twm.start()

        def handle_socket_message(msg):
            # kline 데이터 메시지만 처리
            if msg.get('e') != 'kline':
                return

            kline = msg['k']
            is_candle_closed = kline['x']  # 캔들이 마감되었는지 여부
            open_time = pd.to_datetime(kline['t'], unit='ms', utc=True)
            open_timer = pd.to_datetime(open_time, unit='ms', utc=True)
            #open_time = pd.to_datetime(open_time, unit='ms', utc=True)
            open_time = open_time.tz_convert('Asia/Seoul')


            new_data = {
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v'])
            }
            if open_time in self.data.index:
                # 기존 봉 업데이트 (진행 중인 봉)
                self.data.loc[open_time, 'open'] = new_data['open']
                self.data.loc[open_time, 'high'] = max(self.data.loc[open_time, 'high'], new_data['high'])
                self.data.loc[open_time, 'low'] = min(self.data.loc[open_time, 'low'], new_data['low'])
                self.data.loc[open_time, 'close'] = new_data['close']
                self.data.loc[open_time, 'volume'] += new_data['volume']
            else:
                # 새로운 봉 추가 (이전 봉이 마감되었을 경우)
                self.data.loc[open_time] = new_data
                self.data.sort_index(inplace=True)
            print ('t :', open_time, kline['o'], kline['h'],kline['l'],kline['c'])

        self.twm.start_kline_socket(callback=handle_socket_message, symbol=self.symbol, interval=self.timestep)

    def stop_realtime_socket(self):
        if self.twm:
            self.twm.stop()
            print("웹소켓 연결 종료됨.")

# -----------------------------------------
# 2. IndicatorCalculator: MA, EMA, MACD, STOCHRSI, MASTOCHRSI 계산
# -----------------------------------------
class IndicatorCalculator:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def calculate_all_indicators(self):
        """
        data에 ma, ema, MACD, STOCHRSI, MASTOCHRSI 등의 지표를 추가합니다.
        """
        # 이동평균 (MA)
        for period in [3, 5, 10, 20, 35, 60, 120]:
            self.data[f'ma{period}'] = self.data['close'].rolling(window=period).mean()

        # 지수 이동평균 (EMA)
        for period in [12, 26]:
            self.data[f'ema{period}'] = self.data['close'].ewm(span=period, adjust=False).mean()

        # MACD: DIF, DEA, MACD 히스토그램
        self.data['DIF'] = self.data['ema12'] - self.data['ema26']
        self.data['DEA'] = self.data['DIF'].ewm(span=9, adjust=False).mean()
        self.data['MACD'] = (self.data['DIF'] - self.data['DEA']) * 2

        # STOCHRSI 및 MASTOCHRSI (예시: 14기간 기준, 단순 계산)
        self.data['STOCHRSI'] = self.data['close'].rolling(window=14).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if (x.max()-x.min()) != 0 else 0)
        self.data['MASTOCHRSI'] = self.data['STOCHRSI'].rolling(window=3).mean()

# -----------------------------------------
# 3. SignalGenerator: 매수/매도 신호 생성
# -----------------------------------------
class SignalGenerator:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def generate_signal(self):
        """
        최신 데이터와 지표를 바탕으로 간단한 매수/매도 신호를 생성합니다.
        (예시: MACD와 STOCHRSI 기준)
        """
        if self.data.empty or len(self.data) < 1:
            return None

        latest = self.data.iloc[-1]
        signal = None

        # 간단한 전략 예시
        if latest.get('MACD', 0) > 0 and latest.get('STOCHRSI', 1) < 0.2:
            signal = 'BUY'
        elif latest.get('MACD', 0) < 0 and latest.get('STOCHRSI', 0) > 0.8:
            signal = 'SELL'

        return signal

# -----------------------------------------
# 4. TradeExecutor: Binance 선물 API를 통한 주문 실행
# -----------------------------------------
class TradeExecutor:
    def __init__(self, client: Client, symbol: str):
        self.client = client
        self.symbol = symbol

    def execute_order(self, side: str, quantity: float):
        """
        실제 주문 실행 (시장가 주문)
        """
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            print(f"{side} 주문 실행됨: {order}")
            return order
        except Exception as e:
            print(f"{side} 주문 실행 중 오류 발생:", e)
            return None

# -----------------------------------------
# 5. PlotlyDashboard: 모든 지표를 포함한 대시보드 생성
# -----------------------------------------
class PlotlyDashboard:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.fig = None

    def update_dashboard(self, data: pd.DataFrame, signal: str = None, trade_history: list = None):
        """
        data: OHLC, Volume, MA, EMA, MACD, STOCHRSI 등 모든 지표를 포함한 DataFrame  
        signal: 최신 매매 신호 ('BUY' 또는 'SELL')  
        trade_history: 체결 내역 리스트 (예: {'time': 타임스탬프, 'price': 가격, 'side': 'BUY'/'SELL'})
        """
        # 4행으로 구성된 서브플롯 생성
        self.fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.5, 0.1, 0.2, 0.2],
            specs=[
                [{"type": "candlestick"}],
                [{"type": "bar"}],
                [{"type": "scatter"}],
                [{"type": "scatter"}]
            ]
        )

        # Row 1: 캔들스틱 차트 + MA, EMA 오버레이
        self.fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )

        # MA 선들 추가
        for period in [3, 5, 10, 20, 35, 60, 120]:
            col_name = f"ma{period}"
            if col_name in data.columns:
                self.fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[col_name],
                        mode='lines',
                        name=col_name,
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )

        # EMA 선들 추가
        for period in [12, 26]:
            col_name = f"ema{period}"
            if col_name in data.columns:
                self.fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[col_name],
                        mode='lines',
                        name=col_name,
                        line=dict(width=1, dash='dash')
                    ),
                    row=1, col=1
                )

        # 최신 매매 신호 표시 (마커)
        if signal is not None:
            last_time = data.index[-1]
            last_price = data['close'].iloc[-1]
            marker_symbol = 'triangle-up' if signal == 'BUY' else 'triangle-down'
            marker_color = 'green' if signal == 'BUY' else 'red'
            self.fig.add_trace(
                go.Scatter(
                    x=[last_time],
                    y=[last_price],
                    mode='markers',
                    marker=dict(color=marker_color, size=12, symbol=marker_symbol),
                    name=signal
                ),
                row=1, col=1
            )

        # Row 2: 거래량(Volume) 바 차트
        self.fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                marker_color='blue',
                name='Volume'
            ),
            row=2, col=1
        )

        # Row 3: MACD (DIF, DEA 선 + MACD 히스토그램)
        if all(col in data.columns for col in ['DIF', 'DEA', 'MACD']):
            self.fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['DIF'],
                    mode='lines',
                    name='DIF',
                    line=dict(color='blue', width=1)
                ),
                row=3, col=1
            )
            self.fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['DEA'],
                    mode='lines',
                    name='DEA',
                    line=dict(color='orange', width=1)
                ),
                row=3, col=1
            )
            self.fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['MACD'],
                    name='MACD',
                    marker_color='grey'
                ),
                row=3, col=1
            )

        # Row 4: STOCHRSI 및 MASTOCHRSI
        if 'STOCHRSI' in data.columns:
            self.fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['STOCHRSI'],
                    mode='lines',
                    name='STOCHRSI',
                    line=dict(color='purple', width=1)
                ),
                row=4, col=1
            )
        if 'MASTOCHRSI' in data.columns:
            self.fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MASTOCHRSI'],
                    mode='lines',
                    name='MASTOCHRSI',
                    line=dict(color='red', width=1)
                ),
                row=4, col=1
            )

        # trade_history에 있는 매매 체결 내역 주석(annotation) 추가
        if trade_history:
            for trade in trade_history:
                self.fig.add_annotation(
                    x=trade['time'],
                    y=trade['price'],
                    text=trade['side'],
                    showarrow=True,
                    arrowhead=1,
                    xref='x',
                    yref='y'
                )

        self.fig.update_layout(
            title=f"{self.symbol} Trading Dashboard",
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
        )

    def save_dashboard(self, filename: str = 'dashboard.html'):
        if self.fig:
            self.fig.write_html(filename)
            print(f"대시보드 업데이트 완료: {filename}")

# -----------------------------------------
# 6. TradingBot: 전체 시스템 통합 및 실행 루프
# -----------------------------------------
class TradingBot:
    def __init__(self, symbol: str, timestep: str, api_key: str, api_secret: str):
        self.symbol = symbol
        self.timestep = timestep
        self.client = Client(api_key, api_secret, testnet=True)  # 실거래 시 testnet=False
        self.data_handler = DataHandler(self.client, symbol, timestep)
        self.indicator_calculator = IndicatorCalculator(self.data_handler.data)
        self.signal_generator = SignalGenerator(self.data_handler.data)
        self.trade_executor = TradeExecutor(self.client, symbol)
        self.dashboard = PlotlyDashboard(symbol)
        self.trade_history = []  # 체결된 매매 내역 저장
        self.api_key = api_key
        self.api_secret = api_secret

    def run(self):
        """
        메인 실행 루프:
         - 실시간 데이터는 웹소켓을 통해 업데이트됨
         - 주기적으로 지표 계산, 신호 생성, 주문 실행, 대시보드 업데이트 수행
        """
        print("TradingBot 실행 시작...")
        cnt = 0
        while True:
            cnt = cnt

        if False:
            # 최신 데이터 기반으로 지표 재계산
            self.indicator_calculator.data = self.data_handler.data
            self.indicator_calculator.calculate_all_indicators()

            # 신호 생성
            self.signal_generator.data = self.data_handler.data
            signal = self.signal_generator.generate_signal()
            if signal:
                # 주문 실행 (예시: 고정 수량 0.001)
                order = self.trade_executor.execute_order(signal, quantity=0.001)
                if order:
                    self.trade_history.append({
                        'time': self.data_handler.data.index[-1],
                        'price': self.data_handler.data['close'].iloc[-1],
                        'side': signal
                    })

            # 대시보드 업데이트
            self.dashboard.update_dashboard(self.data_handler.data, signal, self.trade_history)
            self.dashboard.save_dashboard()

            # 1초 주기로 업데이트 (필요시 조정)
            time.sleep(1)

# -----------------------------------------
# 7. 프로그램 시작
# -----------------------------------------
if __name__ == '__main__':
    # 사용자로부터 timestep 옵션을 선택할 수 있음 (예시로 '1m' 사용)
    timestep_option = '1m'
    symbol = 'XRPUSDT'
    #symbol = 'BTCUSDC'

    # Binance API 키 및 시크릿 (환경변수나 안전한 방식으로 관리)
    API_KEY = 'XGldtiQWZP1SB5My8Ecyf1QLDkq5iM2DBKOkp2QBhQk038uRybd80VVKZNMDhr7I'
    API_SECRET = 'yL63MXxfZsGQ2Ceb7CdAJ5KxrTOeqNMx92eCndST0VT3Xr71rNcafOKYjQYrxgUn'

    # TradingBot 인스턴스 생성 및 웹소켓 시작
    bot = TradingBot(symbol, timestep_option, API_KEY, API_SECRET)
    bot.data_handler.start_realtime_socket(API_KEY, API_SECRET)

    try:
        bot.run()
    except KeyboardInterrupt:
        print("프로그램 종료 중...")
        bot.data_handler.stop_realtime_socket()

