import backtrader as bt
from collections import deque
from os.path import join as os_join
from enum import IntEnum
from functools import reduce
import numpy as np

class OrderType(IntEnum):
    BUY = 0
    SELL = 1


def get_order_sign(type_stoploss: OrderType) -> int:
    sign = int(type_stoploss) * 2 - 1
    return sign


def get_order_str(type_order:OrderType) -> str:
    return OrderType.__dict__['_member_names_'][type_order]


def max_sign(sign, *args):
    return sign*max([sign*a for a in args])


class CommInfoFractional(bt.CommissionInfo):
    params = dict(commission=0.0004)  # 0.018% commission 0.00018

    def getsize(self, price, cash):
        '''Returns fractional size for cash operation @price'''
        return (cash / price)

def deepgetattr(obj: object, attr: str, val_default: float = 0.0) -> float:
    """Recurses through an attribute chain to get the ultimate value."""
    try:
        res = reduce(getattr, attr.split('.'), obj)
        return res
    except KeyError:
        return val_default

def print_trade_analysis(analysis, st: bt.Strategy, startcash: float) -> None:
    '''
    Function to print the Technical Analysis results in a nice format.
    '''
    eps = 1e-8
    def get_rounded_analyzer_value(key: str, digits: int = 2) -> float:
        return round(deepgetattr(analysis, key), digits)

    longs_won = get_rounded_analyzer_value('long.won')
    total_longs = get_rounded_analyzer_value('long.total')
    shorts_won = get_rounded_analyzer_value('short.won')
    total_shorts = get_rounded_analyzer_value('short.total')

    long_win_rate = round(longs_won / (total_longs+eps) * 100, 2)
    short_win_rate = round(shorts_won / (total_shorts+eps) * 100, 2)

    total_closed = get_rounded_analyzer_value('total.closed')
    total_won = get_rounded_analyzer_value('won.total')

    win_streak = get_rounded_analyzer_value('streak.won.longest')
    lose_streak = get_rounded_analyzer_value('streak.lost.longest')

    pnl_net = get_rounded_analyzer_value('pnl.net.total')

    win_rate = round((total_won / (total_closed+eps)) * 100, 2)

    total_return = 100 * pnl_net / startcash

    # frequency = deepgetattr(analyzer, 'total.total') / candles_analyzed
    # expectancy = (win_rate/100) * (deepgetattr(st, 'params.risk_reward') - 0.07) - (1 - (win_rate / 100)) - 0.12
    expectancy = (win_rate/100) * (np.mean(deepgetattr(st, '_riskreward_list')) - 0.07) - (1 - (win_rate / 100)) - 0.12
    freq_expectancy = round(total_closed * expectancy, 3)

    total_loss = sum(deepgetattr(st, "_losses_list"))
    total_earning = sum(deepgetattr(st, "_profits_list"))

    # Designate the rows
    h1 = ['Win rate', 'Longs Won', 'Long Win Rate', 'Shorts Won', 'Short Win Rate']
    r1 = [win_rate, '{}/{}'.format(longs_won, total_longs), long_win_rate, '{}/{}'.format(shorts_won, total_shorts),
          short_win_rate]
    h2 = ['Total Trades', 'Win Streak', 'Lose Streak', 'freq_expect', '% Total Return']
    r2 = [total_closed, win_streak, lose_streak, freq_expectancy, total_return]
    # Print the rows
    print_list = [h1, r1, h2, r2]
    row_format = "{:<15}" * (len(h1))
    print("Trade Analysis Results:")
    for row in print_list:
        print(row_format.format(*row))
    print("Total Loss", total_loss)
    print("total_earning", total_earning)
    print("ending balance", startcash + total_earning + total_loss)

# use this one instead: https://www.backtrader.com/docu/order-creation-execution/trail/stoptrail/
class FixedStopTracker(object):
    """
    signals crossing of market entry ticket stop loss
    """
    def __init__(self, typ: OrderType, stoploss_price: float) -> None:
        self.type = typ
        self.stoploss_price = stoploss_price
        self.sign = get_order_sign(typ)
        self._is_active = True

    def __call__(self, actual_price: float) -> bool:
        return (self.sign*actual_price < self.sign*self.stoploss_price) and self._is_active

    def __repr__(self) -> str:
        return f'StopTracker(typ={get_order_str(self.type)}, stoploss_price={self.stoploss_price:.2f})'


class TrailingStopTracker(object):
    """
    signals crossing of trailing stop
    """
    def __init__(self, typ: OrderType, callback_rate: float, activation_price: float) -> None:
        self.type = typ  # the trailing (!!!) ticket type
        self.callback_rate = callback_rate
        self.activation_price = activation_price
        self._is_active = False
        self.sign = get_order_sign(typ)
        self.stoploss_price = stoploss_price

        self._trailing_stop = activation_price * (1 - self.sign * self.callback_rate)

    def update(self, actual_price: float) -> None:
        self._trailing_stop = max_sign(self.sign, self._trailing_stop, actual_price*(1-self.sign*self.callback_rate))

        if not self._is_active:
            self._is_active = self.sign*actual_price > self.sign*self.activation_price

    def __call__(self, actual_price: float) -> bool:
        return (self.sign*actual_price < self.sign*self._trailing_stop) and self._is_active

    def __repr__(self) -> str:
        return f'TrailingStopTracker(typ={get_order_str(self.type)}, callback_rate={self.callback_rate}, ' \
               f'activation_price={self.activation_price:.2f})'

def stoploss_price(data_stream: bt.feeds, order_type: OrderType, entry_price: float, last_atr: float, minimum_stoploss: float) -> float:
    """
    own implementation -> not part of bt.Strategy
    """
    sign = get_order_sign(order_type)
    func = max if order_type == OrderType.SELL else min

    extreme_price_13days = max([data_stream.high[-i] for i in range(2, 15)]) if order_type == OrderType.SELL else \
                           min([data_stream.low[-i] for i in range(2, 15)])

    stoploss = entry_price * (1 + sign * minimum_stoploss)

    dist = abs(entry_price - extreme_price_13days)

    if dist > 2 * last_atr:
        stop_loss_ATR = entry_price + sign * last_atr
    else:
        stop_loss_ATR = extreme_price_13days + sign * 0.5 * last_atr
    return func(stop_loss_ATR, stoploss)


class RiskReward(object):
    def __init__(self, price_stoploss: float, price_entry: float = 0.0) -> None:
        self.price_entry = price_entry  # from order creation
        self.price_stoploss = price_stoploss  # from order creation

    # only needed if entry price should be added at order execution rather than order creation
    def set_execution_price(self, price_entry: float) -> None:
        self.price_entry = price_entry

    def __call__(self, price_exit):  # price_exit from market exit execution
        assert self.price_entry > 0
        return abs((price_exit - self.price_entry)/(self.price_entry - self.price_stoploss))


class TrailingStop_v2(bt.Strategy):
    """
    this strategy buys or sells on ma crossing and then finishes the trade either on stop_loss or trailing_stop.
    """
    params = (
        ('maperiod', 15),
        ('taperiod', 15),
        ('risk_reward', 3.3),  #factor
        ('callback_rate', 0.05),  #pct
        ('minimum_stoploss', 0.007),  #pct
        ('risk', 0.001),  # pct
        ('historic_maxlen', 10)  # last n historic prices before trade to display
    )

    # class instance, since collection across all trades necessary
    _losses_list = []
    _profits_list = []
    _riskreward_list = []

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.trailingstop = None
        self.lossstop = None
        self.riskreward = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod)
        self.atr = bt.indicators.AverageTrueRange(self.datas[0], period=self.params.taperiod)

        self._historicprices_queue = deque([], maxlen=self.params.historic_maxlen)  # 10 last price tuples

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            # self.log('\n'.join([str(s) for s in self._historicprices_queue]))
            pnl = order.executed.pnl
            if pnl > 0:
                self._profits_list.append(pnl)
            else:
                self._losses_list.append(pnl)

            # we just exited the market, add risk_reward achieved
            # if not self.position:
            if getattr(self.order, 'name', False) and self.order.name == 'market_exit':  # equivalent to above
                # add the final reward
                self._riskreward_list.append(self.riskreward(price_exit=order.executed.price))
            # else:
            #    self.riskreward.set_execution_price(self.order.executed.price)

            order_type_trailing = None
            if order.isbuy():
                order_type_trailing =OrderType.SELL
                str1 = 'BUY'
            elif order.issell():
                order_type_trailing =OrderType.BUY
                str1 = 'SELL'

            if order_type_trailing is not None:

                self.log(
                    '%s EXECUTED, Price:  %.2f, Cost: %.2f, Comm %.2f' %
                    (str1,
                     order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                sign = get_order_sign(order_type_trailing)
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm

                # activate the trailingstop tracker only if we have a position, that is we are in the market
                if self.position:
                    sl = stoploss_price(self.data, order_type=1-order_type_trailing, entry_price=self.buyprice, last_atr=self.atr[-1],
                                        minimum_stoploss=self.params.minimum_stoploss)
                    self.log(
                        f'sl calc -- data[0]: {self.data[0]}, order_type ={1 - order_type_trailing}, entry_price={self.buyprice}, '
                        f'last_atr: {self.atr[-1]} => sl: {sl}')
                    target_price = self.buyprice + self.params.risk_reward * (-sl + self.buyprice)
                    assert sign*self.buyprice <= sign*target_price
                    self.trailingstop = TrailingStopTracker(typ=order_type_trailing, callback_rate=self.params.callback_rate,
                                                            activation_price=0.5*(self.buyprice + target_price))

                    # self.riskreward = RiskReward(price_entry=order.executed.price, price_stoploss=sl)

                    self.log(f'... and activating "{self.trailingstop}" with price_trailstop "{self.trailingstop._trailing_stop:.2f}" @ target_price: {target_price:.2f}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))
        self._historicprices_queue.clear()

        self.trailingstop = None
        self.lossstop = None

    def next(self):
        # Simply log the closing price of the series from the reference
        #self.log('Close, %.2f Position, %.2f' % (self.dataclose[0], self.position.size))
        pts = None
        if self.trailingstop:
            self.trailingstop.update(self.dataclose[0])
            pts = self.trailingstop._trailing_stop
        self._historicprices_queue.append({"date": self.datas[0].datetime.date(0).isoformat(),
                                           'close_price': self.dataclose[0],
                                           'position': self.position.size,
                                           'price_trailstop': pts})

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # If we are not in the market we watch the MA signal to get into it at the right point in time
        if not self.position:
            order_type = None
            if self.dataclose[0] > self.sma[0] and self.dataclose[-1] < self.sma[-1]:
                order_type = OrderType.BUY

            elif self.dataclose[0] < self.sma[0] and self.dataclose[-1] > self.sma[-1]:
                order_type = OrderType.SELL

            # Keep track of the created order to avoid a 2nd order
            # self.order = self.buy()  # buys always 1 position
            # self.order = self.order_target_size(target=20)  # equivalent, unless position was already = 1
            if order_type is not None and not np.isnan(self.atr[-1]):
                print('\n'.join([str(s) for s in self._historicprices_queue][-len(self):]))
                risk_amount = self.params.risk * self.broker.getvalue()
                sl = stoploss_price(self.data, order_type=order_type, entry_price=self.dataclose[0], last_atr=self.atr[-1],
                                    minimum_stoploss=self.params.minimum_stoploss)
                self.log(f'sl calc -- data[0]: {self.data[0]}, order_type ={order_type}, entry_price={self.dataclose[0]}, '
                         f'last_atr: {self.atr[-1]} => sl: {sl}')
                target = risk_amount / (self.dataclose[0] - sl)
                self.order = self.order_target_size(target=target)  # equivalent, unless position was already = 1

                # enable the stoploss tracker
                self.lossstop = FixedStopTracker(OrderType.SELL if order_type == OrderType.BUY else OrderType.BUY, stoploss_price=sl)

                # self.riskreward = RiskReward(price_stoploss=sl)
                # setting the risk first
                self.riskreward = RiskReward(price_entry=self.dataclose[0], price_stoploss=sl)

                self.log(f'{get_order_str(order_type)} (={self.order.ordtype}) CREATE, close_price: %.2f, stop_loss: %.2f' % (self.dataclose[0], sl))
                self.log(f'... and activating "{self.lossstop}" ')

        # if we are in the market we wait until our trailingstop tracker signals us to act (buy or sell). In that moment we place an order.
        else:
            if self.trailingstop(self.dataclose[0]) or self.lossstop(self.dataclose[0]):
                print('\n'.join([str(s) for s in self._historicprices_queue][-len(self):]))
                # Keep track of the created order to avoid a 2nd order
                # self.order = self.sell() # sell always 1 position
                self.order = self.order_target_size(target=0)
                self.order.name = 'market_exit'
                # self.order = self.order_target_value(target=0)

                # which trigger was activated?
                str1 = 'TRAILING STOP' if self.trailingstop(self.dataclose[0]) else 'STOPLOSS'
                self.log(f'{get_order_str(self.order.ordtype)} CREATE from {str1}, %.2f' % self.dataclose[0])





# FROM main.py

if __name__ == '__main__':

    data_sim = None
    params_sim = None

    # adapt the path to your directory
    pfad = '/Users/thomas/data/upwork/trailingstop'
    symbol = 'Upwork_15m'
    datafile = 'data.csv'
    data_csv = bt.feeds.GenericCSVData(dataname=datafile,
                                       # fromdate=datetime(2020, 12, 23),
                                       # todate=datetime(2020, 11, 24),
                                       openinterest=-1,
                                       timeframe=bt.TimeFrame.Minutes,
                                       dtformat=('%Y-%m-%d %H:%M:%S'))

    # if you want to run with stoploss additionally just set to True
    params_csv = dict(callback_rate=0.04, risk_reward=3, maperiod=200, taperiod=200, historic_maxlen=5)


    # choose here simulation or bitcoin
    data, params = [(data_sim, params_sim), (data_csv, params_csv)][1]

    startcash = 1_000_000
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(startcash)
    cerebro.adddata(data)
    # cerebro.addstrategy(TestStrategy)
    cerebro.addstrategy(TrailingStop_v2, **params)  # want immediate activation of trailing_stop
    cerebro.broker.addcommissioninfo(CommInfoFractional())
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
    strategies = cerebro.run()

    print_trade_analysis(analysis=strategies[0].analyzers.ta.get_analysis(), st=TrailingStop_v2, startcash=startcash)
